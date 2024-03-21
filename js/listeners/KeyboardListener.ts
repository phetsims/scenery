// Copyright 2022-2024, University of Colorado Boulder

/**
 * A listener for general keyboard input. Specify the keys with a `keys` option in a readable format that looks like
 * this: [ 'shift+t', 'alt+shift+r' ]
 *
 * - Each entry in the array represents a "group" of keys.
 * - '+' separates each key in a single group.
 * - The keys leading up to the last key in the group are considered "modifier" keys. The last key in the group needs
 *   to be pressed while the modifier keys are down.
 * - The order modifier keys are pressed does not matter for firing the callback.
 *
 * In the above example "shift+t" OR "alt+shift+r" will fire the callback when pressed.
 *
 * An example usage would like this:
 *
 * this.addInputListener( new KeyboardListener( {
 *   keys: [ 'a+b', 'a+c', 'shift+arrowLeft', 'alt+g+t', 'ctrl+3', 'alt+ctrl+t' ],
 *   callback: ( event, keysPressed, listener ) => {
 *     if ( keysPressed === 'a+b' ) {
 *       console.log( 'you just pressed a+b!' );
 *     }
 *     else if ( keysPressed === 'a+c' ) {
 *       console.log( 'you just pressed a+c!' );
 *     }
 *     else if ( keysPressed === 'alt+g+t' ) {
 *       console.log( 'you just pressed alt+g+t' );
 *     }
 *     else if ( keysPressed === 'ctrl+3' ) {
 *       console.log( 'you just pressed ctrl+3' );
 *     }
 *     else if ( keysPressed === 'shift+arrowLeft' ) {
 *       console.log( 'you just pressed shift+arrowLeft' );
 *     }
 *   }
 * } ) );
 *
 * By default the callback will fire when the last key is pressed down. See additional options for firing on key
 * up or other press and hold behavior.
 *
 *  -- Overlapping Listener Behavior --
 * Scenery's input system will detect if there are multiple KeyboardListeners that will fire for the same keys. With
 * the default behavior, the listener with the most specific keys will fire. If the keys of one listener are a subset
 * of another listener, only the listener with more specific keys will fire its callback. For example, if one listener
 * has 'shift+t' and another listener has 't', the listener with 'shift+t' will fire and the listener with 't' will
 * not. If two listeners use the exact same keys, an error will be thrown. This behavior can be controlled with
 * the `listenerOverlapBehavior` option. See the documentation for that option for more information.
 *
 * **Potential Pitfall!**
 * The callback is only called if exactly the keys in a group are pressed. If you need to listen to a modifier key,
 * you must include it in the keys array. For example if you add a listener for 'tab', you must ALSO include
 * 'shift+tab' in the array to observe 'shift+tab' presses. If you provide 'tab' alone, the callback will not fire
 * if 'shift' is also pressed.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import CallbackTimer from '../../../axon/js/CallbackTimer.js';
import optionize from '../../../phet-core/js/optionize.js';
import { EnglishKey, EnglishStringToCodeMap, FocusManager, globalKeyStateTracker, scenery, SceneryEvent, TInputListener } from '../imports.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import EnabledComponent, { EnabledComponentOptions } from '../../../axon/js/EnabledComponent.js';

// NOTE: The typing for ModifierKey and OneKeyStroke is limited TypeScript, there is a limitation to the number of
//       entries in a union type. If that limitation is not acceptable remove this typing. OR maybe TypeScript will
//       someday support regex patterns for a type. See https://github.com/microsoft/TypeScript/issues/41160
// If we run out of union space for template strings, consider the above comment or remove some from the type.
type ModifierKey = 'q' | 'w' | 'e' | 'r' | 't' | 'y' | 'u' | 'i' | 'o' | 'p' | 'a' | 's' | 'd' |
  'f' | 'g' | 'h' | 'j' | 'k' | 'l' | 'z' | 'x' | 'c' |
  'v' | 'b' | 'n' | 'm' | 'ctrl' | 'alt' | 'shift' | 'tab';

// Allowed keys are the keys of the EnglishStringToCodeMap.
type AllowedKeys = keyof typeof EnglishStringToCodeMap;

export type OneKeyStroke = `${AllowedKeys}` |
  `${ModifierKey}+${AllowedKeys}` |
  `${ModifierKey}+${ModifierKey}+${AllowedKeys}`;
// These combinations are not supported by TypeScript: "TS2590: Expression produces a union type that is too complex to
// represent." See above note and https://github.com/microsoft/TypeScript/issues/41160#issuecomment-1287271132.
// `${AllowedKeys}+${AllowedKeys}+${AllowedKeys}+${AllowedKeys}`;
// type KeyCombinations = `${OneKeyStroke}` | `${OneKeyStroke},${OneKeyStroke}`;

// Controls when the callback listener fires.
// - 'up': Callbacks fire on release of keys.
// - 'down': Callbacks fire on press of keys.
// - 'both': Callbacks fire on both press and release of keys.
type ListenerFireTrigger = 'up' | 'down' | 'both';

// Controls how the listener behaves when another listener is found that will fire for the same keys.
// - 'most_specific': The listener with the most specific keys will fire. If the keys are a subset of another listener,
//   the listener will be deferred, and the callback will not fire. This is the default behavior.
// - 'allow': The listener will fire even if another listener has more specific keys, or if another listener
//   uses the exact same keys.
// - 'error': The listener will throw an error if another listener has more specific keys, or if another listener
type ListenerOverlapBehavior = 'most_specific' | 'allow' | 'error';

type KeyGroupWithListener = {
  listener: KeyboardListener<OneKeyStroke[]>;
  keyGroup: KeyGroup<OneKeyStroke[]>;
};

// A global collection of all deferred KeyboardListeners. These are listeners that will not fire because another
// listener has more specific keys. Listeners are added to this map if there is a detected overlap with another
// listener on key down. When a key is released, the listener will be undeferred and removed from the map.
const deferredKeyboardListenersMap = new Map<string, KeyboardListener<OneKeyStroke[]>[]>();

type SelfOptions<Keys extends readonly OneKeyStroke[ ]> = {

  // The keys that need to be pressed to fire the callback. In a form like `[ 'shift+t', 'alt+shift+r' ]`. See top
  // level documentation for more information and an example of providing keys.
  keys: Keys;

  // If true, the listener will fire for keys regardless of where focus is in the document. Use this when you want
  // to add some key press behavior that will always fire no matter what the event target is. If this listener
  // is added to a Node, it will only fire if the Node (and all of its ancestors) are visible with inputEnabled: true.
  // More specifically, this uses `globalKeyUp` and `globalKeyDown`. See definitions in Input.ts for more information.
  global?: boolean;

  // If true, this listener is fired during the 'capture' phase. Only relevant for `global` key events.
  // When a listener uses capture, the callbacks will be fired BEFORE the dispatch through the scene graph
  // (very similar to DOM's addEventListener with `useCapture` set to true - see
  // https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener).
  capture?: boolean;

  // If true, all SceneryEvents that trigger this listener (keydown and keyup) will be `handled` (no more
  // event bubbling). See `manageEvent` for more information.
  handle?: boolean;

  // If true, all SceneryEvents that trigger this listener (keydown and keyup) will be `aborted` (no more
  // event bubbling, no more listeners fire). See `manageEvent` for more information.
  abort?: boolean;

  // Called when the listener detects that the set of keys are pressed.
  callback?: ( event: SceneryEvent<KeyboardEvent> | null, keysPressed: Keys[number], listener: KeyboardListener<Keys> ) => void;

  // Called when the listener is cancelled/interrupted.
  cancel?: ( listener: KeyboardListener<Keys> ) => void;

  // Called when the listener target receives focus.
  focus?: ( listener: KeyboardListener<Keys> ) => void;

  // Called when the listener target loses focus.
  blur?: ( listener: KeyboardListener<Keys> ) => void;

  // When true, the listener will fire continuously while keys are held down, at the following intervals.
  fireOnHold?: boolean;

  // If fireOnHold true, this is the delay in (in milliseconds) before the callback is fired continuously.
  fireOnHoldDelay?: number;

  // If fireOnHold true, this is the interval (in milliseconds) that the callback fires after the fireOnHoldDelay.
  fireOnHoldInterval?: number;

  // Possible input types that decide when callbacks of the listener fire in response to input. See
  // ListenerFireTrigger type documentation.
  listenerFireTrigger?: ListenerFireTrigger;

  // Controls how the listener behaves when another listener is found that will fire for the same keys. See
  // ListenerOverlapBehavior type documentation.
  listenerOverlapBehavior?: ListenerOverlapBehavior;
};

export type KeyboardListenerOptions<Keys extends readonly OneKeyStroke[]> = SelfOptions<Keys> & EnabledComponentOptions;

type KeyGroup<Keys extends readonly OneKeyStroke[]> = {

  // All must be pressed fully before the key is pressed to activate the command.
  modifierKeys: string[][];

  // Contains the triggering key for the listener. One of these keys must be pressed to activate callbacks.
  keys: string[];

  // All of the keys that must be pressed for the callback to fire (modifier keys up to the leading key)
  allKeyCodes: string[];

  // All keys in this KeyGroup using the readable form
  naturalKeys: Keys[number];

  // A callback timer for this KeyGroup to support press and hold timing and callbacks
  timer: CallbackTimer | null;
};

class KeyboardListener<Keys extends readonly OneKeyStroke[]> extends EnabledComponent implements TInputListener {

  // The function called when a KeyGroup is pressed (or just released). Provides the SceneryEvent that fired the input
  // listeners and this the keys that were pressed from the active KeyGroup. The event may be null when using
  // fireOnHold or in cases of cancel or interrupt. A reference to the listener is provided for other state.
  private readonly _callback: ( event: SceneryEvent<KeyboardEvent> | null, keysPressed: Keys[number], listener: KeyboardListener<Keys> ) => void;

  // The optional function called when this listener is cancelled.
  private readonly _cancel: ( listener: KeyboardListener<Keys> ) => void;

  // The optional function called when this listener's target receives focus.
  private readonly _focus: ( listener: KeyboardListener<Keys> ) => void;

  // The optional function called when this listener's target loses focus.
  private readonly _blur: ( listener: KeyboardListener<Keys> ) => void;

  // When callbacks are fired in response to input. Could be on keys pressed down, up, or both.
  private readonly _listenerFireTrigger: ListenerFireTrigger;

  // Does the listener fire the callback continuously when keys are held down?
  private readonly _fireOnHold: boolean;

  // (scenery-internal) All the KeyGroups of this listener from the keys provided in natural language.
  public readonly _keyGroups: KeyGroup<Keys>[];

  // All the KeyGroups that are currently firing
  private readonly _activeKeyGroups: KeyGroup<Keys>[];

  // True when keys are pressed down. If listenerFireTrigger is 'both', you can look at this in your callback to
  // determine if keys are pressed or released.
  public keysDown: boolean;

  // Timing variables for the CallbackTimers.
  private readonly _fireOnHoldDelay: number;
  private readonly _fireOnHoldInterval: number;

  public readonly listenerOverlapBehavior: ListenerOverlapBehavior;

  // (scenery-internal) - Scenery has found that another KeyboardListener will fire for the same keys. It will
  // defer this listener if the other key has more specific keys.
  public _deferred = false;

  // see options documentation
  public readonly global: boolean;
  private readonly _handle: boolean;
  private readonly _abort: boolean;

  private readonly _windowFocusListener: ( windowHasFocus: boolean ) => void;

  public constructor( providedOptions: KeyboardListenerOptions<Keys> ) {
    const options = optionize<KeyboardListenerOptions<Keys>, SelfOptions<Keys>, EnabledComponentOptions>()( {
      callback: _.noop,
      cancel: _.noop,
      focus: _.noop,
      blur: _.noop,
      global: false,
      capture: false,
      handle: false,
      abort: false,
      listenerFireTrigger: 'down',
      fireOnHold: false,
      fireOnHoldDelay: 400,
      fireOnHoldInterval: 100,
      listenerOverlapBehavior: 'most_specific'
    }, providedOptions );

    super( providedOptions );

    this._callback = options.callback;
    this._cancel = options.cancel;
    this._focus = options.focus;
    this._blur = options.blur;

    this._listenerFireTrigger = options.listenerFireTrigger;
    this._fireOnHold = options.fireOnHold;
    this._fireOnHoldDelay = options.fireOnHoldDelay;
    this._fireOnHoldInterval = options.fireOnHoldInterval;
    this.listenerOverlapBehavior = options.listenerOverlapBehavior;

    this._activeKeyGroups = [];

    this.keysDown = false;

    this.global = options.global;
    this._handle = options.handle;
    this._abort = options.abort;

    // convert the provided keys to data that we can respond to with scenery's Input system
    this._keyGroups = this.convertKeysToKeyGroups( options.keys );

    // Assign listener and capture to this, implementing TInputListener
    ( this as unknown as TInputListener ).listener = this;
    ( this as unknown as TInputListener ).capture = options.capture;

    this._windowFocusListener = this.handleWindowFocusChange.bind( this );
    FocusManager.windowHasFocusProperty.link( this._windowFocusListener );

    // interrupt when the listener is disabled
    this.enabledProperty.lazyLink( this.onEnabledPropertyChange.bind( this ) );
  }

  /**
   * Mostly required to fire with CallbackTimer since the callback cannot take arguments.
   */
  public fireCallback( event: SceneryEvent<KeyboardEvent> | null, keyGroup: KeyGroup<Keys> ): void {
    if ( !this._deferred ) {
      this._callback( event, keyGroup.naturalKeys, this );
    }
  }

  /**
   * Responding to a keydown event, update active KeyGroups and potentially fire callbacks and start CallbackTimers.
   */
  private handleKeyDown( event: SceneryEvent<KeyboardEvent> ): void {
    if ( !this.canPress() ) {
      return;
    }

    if ( this._listenerFireTrigger === 'down' || this._listenerFireTrigger === 'both' ) {

      // modifier keys can be pressed in any order but the last key in the group must be pressed last
      this._keyGroups.forEach( keyGroup => {

        if ( !this._activeKeyGroups.includes( keyGroup ) ) {
          if ( this.areKeysDownForListener( keyGroup ) &&
               keyGroup.keys.includes( globalKeyStateTracker.getLastKeyDown()! ) ) {

            this._activeKeyGroups.push( keyGroup );

            this.keysDown = true;

            // reserve the event for this listener, disabling more 'global' input listeners such as
            // those for pan and zoom (this is similar to DOM event.preventDefault).
            event.pointer.reserveForKeyboardDrag();

            if ( keyGroup.timer ) {
              keyGroup.timer.start();
            }

            this.fireCallback( event, keyGroup );
          }
        }
      } );
    }

    this.manageEvent( event );
  }

  /**
   * If there are any active KeyGroup firing stop and remove if KeyGroup keys are no longer down. Also, potentially
   * fires a KeyGroup callback if the key that was released has all other modifier keys down.
   */
  private handleKeyUp( event: SceneryEvent<KeyboardEvent> ): void {
    if ( !this.canPress() ) {
      return;
    }

    if ( this._activeKeyGroups.length > 0 ) {
      this._activeKeyGroups.forEach( ( activeKeyGroup, index ) => {
        if ( !this.areKeysDownForListener( activeKeyGroup ) ) {
          if ( activeKeyGroup.timer ) {
            activeKeyGroup.timer.stop( false );
          }
          this._activeKeyGroups.splice( index, 1 );
        }
      } );
    }

    if ( this._listenerFireTrigger === 'up' || this._listenerFireTrigger === 'both' ) {
      const eventCode = KeyboardUtils.getEventCode( event.domEvent )!;

      // Screen readers may send key events with no code for unknown reasons, we need to be graceful when that
      // happens, see https://github.com/phetsims/scenery/issues/1534.
      if ( eventCode ) {
        this._keyGroups.forEach( keyGroup => {
          if ( this.areModifierKeysDownForListener( keyGroup ) &&
               keyGroup.keys.includes( eventCode ) ) {
            this.keysDown = false;
            this.fireCallback( event, keyGroup );
          }
        } );
      }
    }

    this.manageEvent( event );
  }

  /**
   * Returns an array of KeyboardEvent.codes from the provided key group that are currently pressed down.
   */
  private getDownModifierKeys( keyGroup: KeyGroup<Keys> ): string[] {

    // Remember, this is a 2D array. The inner array is the list of 'equivalent' keys to be pressed for the required
    // modifier key. For example [ 'shiftLeft', 'shiftRight' ]. If any of the keys in that inner array are pressed,
    // that set of modifier keys is considered pressed.
    const modifierKeysCollection = keyGroup.modifierKeys;

    // The list of modifier keys that are actually pressed
    const downModifierKeys: string[] = [];
    modifierKeysCollection.forEach( modifierKeys => {
      for ( const modifierKey of modifierKeys ) {
        if ( globalKeyStateTracker.isKeyDown( modifierKey ) ) {
          downModifierKeys.push( modifierKey );

          // One modifier key from this inner set is down, stop looking
          break;
        }
      }
    } );

    return downModifierKeys;
  }

  /**
   * Returns true if keys are pressed such that the listener should fire. In order to fire, all modifier keys
   * should be down and the final key of the group should be down.
   */
  private areKeysDownForListener( keyGroup: KeyGroup<Keys> ): boolean {
    const downModifierKeys = this.getDownModifierKeys( keyGroup );

    // modifier keys are down if one key from each inner array is down
    const areModifierKeysDown = downModifierKeys.length === keyGroup.modifierKeys.length;

    // The final key of the group is down if any of them are pressed
    const finalDownKey = keyGroup.keys.find( key => globalKeyStateTracker.isKeyDown( key ) );

    return areModifierKeysDown && !!finalDownKey;
  }

  /**
   * Returns true if the modifier keys of the provided key group are currently down.
   */
  private areModifierKeysDownForListener( keyGroup: KeyGroup<Keys> ): boolean {
    const downModifierKeys = this.getDownModifierKeys( keyGroup );

    // modifier keys are down if one key from each inner array is down
    return downModifierKeys.length === keyGroup.modifierKeys.length;
  }

  /**
   * Returns whether a key press can be started for the listener.
   *
   * NOTE: Could take the event in the future if state of the event determines whether the listener can press.
   */
  private canPress(): boolean {
    return this.enabledProperty.value;
  }

  /**
   * Interrupt if the listener is disabled.
   */
  private onEnabledPropertyChange( enabled: boolean ): void {
    !enabled && this.interrupt();
  }

  /**
   * In response to every SceneryEvent, handle and/or abort depending on listener options. This cannot be done in
   * the callbacks because press-and-hold behavior triggers many keydown events. We need to handle/abort each, not
   * just the event that triggered the callback. Also, callbacks can be called without a SceneryEvent from the
   * CallbackTimer.
   */
  private manageEvent( event: SceneryEvent<KeyboardEvent> ): void {
    this._handle && event.handle();
    this._abort && event.abort();
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the keydown event when not
   * added to the global key events. Target will be the Node, Display, or Pointer this listener was added to.
   */
  public keydown( event: SceneryEvent<KeyboardEvent> ): void {
    if ( !this.global ) {
      this.handleKeyDown( event );
    }
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the keyup event when not
   * added to the global key events. Target will be the Node, Display, or Pointer this listener was added to.
   */
  public keyup( event: SceneryEvent<KeyboardEvent> ): void {
    if ( !this.global ) {
      this.handleKeyUp( event );
    }
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the global keydown event.
   * Event has no target.
   */
  public globalkeydown( event: SceneryEvent<KeyboardEvent> ): void {
    if ( this.global ) {
      this.handleKeyDown( event );
    }
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the global keyup event.
   * Event has no target.
   */
  public globalkeyup( event: SceneryEvent<KeyboardEvent> ): void {
    if ( this.global ) {
      this.handleKeyUp( event );
    }
  }

  /**
   * Work to be done on both cancel and interrupt.
   */
  private handleCancel(): void {
    this.clearActiveKeyGroups();
    this._cancel( this );
  }

  /**
   * When the window loses focus, cancel.
   */
  private handleWindowFocusChange( windowHasFocus: boolean ): void {
    if ( !windowHasFocus ) {
      this.handleCancel();
    }
  }

  /**
   * Part of the scenery listener API. On cancel, clear active KeyGroups and stop their behavior.
   */
  public cancel(): void {
    this.handleCancel();
  }

  /**
   * Part of the scenery listener API. Clear active KeyGroups and stop their callbacks.
   */
  public interrupt(): void {
    this.handleCancel();
  }

  /**
   * Interrupts and resets the listener on blur so that state is reset and active keyGroups are cleared.
   * Public because this is called with the scenery listener API. Do not call this directly.
   */
  public focusout( event: SceneryEvent ): void {
    this.interrupt();

    // Optional work to do on blur.
    this._blur( this );
  }

  /**
   * Public because this is called through the scenery listener API. Do not call this directly.
   */
  public focusin( event: SceneryEvent ): void {

    // Optional work to do on focus.
    this._focus( this );
  }

  /**
   * Dispose of this listener by disposing of any Callback timers. Then clear all KeyGroups.
   */
  public override dispose(): void {
    super.dispose();

    this._keyGroups.forEach( activeKeyGroup => {
      activeKeyGroup.timer && activeKeyGroup.timer.dispose();
    } );
    this._keyGroups.length = 0;

    FocusManager.windowHasFocusProperty.unlink( this._windowFocusListener );
  }

  /**
   * Clear the active KeyGroups on this listener. Stopping any active groups if they use a CallbackTimer.
   */
  private clearActiveKeyGroups(): void {
    this._activeKeyGroups.forEach( activeKeyGroup => {
      activeKeyGroup.timer && activeKeyGroup.timer.stop( false );
    } );

    this._activeKeyGroups.length = 0;
  }

  /**
   * Converts the provided keys into a collection of KeyGroups to easily track what keys are down. For example,
   * will take a string that defines the keys for this listener like 'a+c|1+2+3+4|shift+leftArrow' and return an array
   * with three KeyGroups, one describing 'a+c', one describing '1+2+3+4' and one describing 'shift+leftArrow'.
   */
  private convertKeysToKeyGroups( keys: Keys ): KeyGroup<Keys>[] {

    const keyGroups = keys.map( naturalKeys => {

      // all of the keys in this group in an array
      const groupKeys = naturalKeys.split( '+' );
      assert && assert( groupKeys.length > 0, 'no keys provided?' );

      const naturalKey = groupKeys.slice( -1 )[ 0 ] as AllowedKeys;
      const keys = EnglishStringToCodeMap[ naturalKey ]!;
      assert && assert( keys, `Codes were not found, do you need to add it to EnglishStringToCodeMap? ${naturalKey}` );

      let modifierKeys: string[][] = [];
      if ( groupKeys.length > 1 ) {
        const naturalModifierKeys = groupKeys.slice( 0, groupKeys.length - 1 ) as ModifierKey[];
        modifierKeys = naturalModifierKeys.map( naturalModifierKey => {
          const modifierKeys = EnglishStringToCodeMap[ naturalModifierKey ]!;
          assert && assert( modifierKeys, `Key not found, do you need to add it to EnglishStringToCodeMap? ${naturalModifierKey}` );
          return modifierKeys;
        } );
      }

      // Set up the timer for triggering callbacks if this listener supports press and hold behavior
      const timer = this._fireOnHold ? new CallbackTimer( {
        callback: () => this.fireCallback( null, keyGroup ),
        delay: this._fireOnHoldDelay,
        interval: this._fireOnHoldInterval
      } ) : null;

      const keyGroup: KeyGroup<Keys> = {
        keys: keys,
        modifierKeys: modifierKeys,
        naturalKeys: naturalKeys,
        allKeyCodes: [ ...modifierKeys.flat(), ...keys ],
        timer: timer
      };
      return keyGroup;
    } );

    return keyGroups;
  }

  /**
   * Look for overlapping keys in the provided listeners. This is used by Input.ts when it is time to respond to
   * keyboard events. Input.ts provides all of the KeyboardListeners that may respond to user input. Behavior
   * of the listeners is then controlled by the listenerOverlapBehavior option.
   *
   * To decide which listenerOverlapBehavior to use, this logic is used:
   * - If either listener has 'error' behavior, the 'error' behavior is used.
   * - Else if either listener has 'allow' behavior, the 'allow' behavior is used.
   * - Otherwise, use 'most_specific' behavior (would be on both listeners)
   *
   * Then, the function behaves like this depending on the listenerOverlapBehavior:
   * - 'most_specific': If a listener uses a subset of another listener's keys, the listener will be deferred and
   *    only the other listener will fire. If any two listeners use the exact same keys, an error will be thrown.
   * - 'error': If any two listeners use the same keys or if one listener uses a subset of another listener's keys,
   *   an error will be thrown.
   * - 'allow': All listeners will fire, even if they use the same keys or if one listener uses a subset of another
   *   listener's keys.
   */
  public static inspectKeyboardListeners( keyboardListeners: KeyboardListener<OneKeyStroke[]>[], event: KeyboardEvent ): void {

    // Collect KeyGroups with their listeners so can easily look up the listener when iterating through used keys.
    const naturalKeysWithListener = keyboardListeners.reduce( ( accumulator: KeyGroupWithListener[], listener: KeyboardListener<OneKeyStroke[]> ) => {
      const keyGroups = listener._keyGroups;
      keyGroups.forEach( keyGroup => {
        accumulator.push( {
          listener: listener,
          keyGroup: keyGroup
        } );
      } );
      return accumulator;
    }, [] );

    // Compares each listener with every other, only visiting each pair once
    for ( let i = 0; i < naturalKeysWithListener.length; i++ ) {
      for ( let j = i + 1; j < naturalKeysWithListener.length; j++ ) {
        const objectA = naturalKeysWithListener[ i ];
        const objectB = naturalKeysWithListener[ j ];

        // Convert keys from "readable" string to an array so that we can easily compare them.
        const aSplitKeys = objectA.keyGroup.naturalKeys.split( '+' );
        const bSplitKeys = objectB.keyGroup.naturalKeys.split( '+' );

        const [ shorterObject, longerObject ] = aSplitKeys.length < bSplitKeys.length ? [ objectA, objectB ] : [ objectB, objectA ];
        const [ shorterKeys, longerKeys ] = aSplitKeys.length < bSplitKeys.length ? [ aSplitKeys, bSplitKeys ] : [ bSplitKeys, aSplitKeys ];

        // If the shorter keys are a subset of the longer keys, there is an overlap that we need to handle. The
        // listener with less specific keys may be deferred based on the behavior.
        if ( KeyboardUtils.isArraySubset( shorterKeys, longerKeys ) ) {
          if ( longerObject.listener.areKeysDownForListener( longerObject.keyGroup ) ) {

            const subsetBehavior = shorterObject.listener.listenerOverlapBehavior;
            const supersetBehavior = longerObject.listener.listenerOverlapBehavior;

            const eitherError = subsetBehavior === 'error' || supersetBehavior === 'error';
            const eitherAllow = subsetBehavior === 'allow' || supersetBehavior === 'allow';

            if ( eitherError ) {
              assert && assert( false, `Overlap detected in KeyboardListeners. ${shorterKeys} key(s) are a subset of the ${longerKeys} key(s)` );
            }
            else if ( eitherAllow ) {

              // 'allow' behavior, nothing to do
            }
            else {

              // 'most_specific' behavior - if both listeners use the same keys, an error will be thrown. Otherwise,
              // the listener with less specific keys will be deferred.
              assert && assert(
                shorterKeys.length !== longerKeys.length,
                'Overlap detected in KeyboardListeners. The keys are the same. Use listenerOverlapBehavior: "allow" or change keys.'
              );

              // Both listeners have 'most_specific' behavior, defer the listener with less specific keys
              shorterObject.listener._deferred = true;

              const keyCode = event.code;
              const listener = shorterObject.listener;
              if ( deferredKeyboardListenersMap.has( keyCode ) ) {
                deferredKeyboardListenersMap.get( keyCode )!.push( listener );
              }
              else {
                deferredKeyboardListenersMap.set( keyCode, [ listener ] );
              }
            }
          }
        }
      }
    }
  }

  /**
   * Undefer all listeners that are deferred for the provided keyCode. This should be called when a key is released.
   */
  public static undeferKeyboardListeners( keyCode: string ): void {
    if ( deferredKeyboardListenersMap.has( keyCode ) ) {
      deferredKeyboardListenersMap.get( keyCode )!.forEach( listener => { listener._deferred = false; } );
      deferredKeyboardListenersMap.delete( keyCode );
    }

    // Check the map when all keys have been released - if all keys are released and there are deferred listeners.
    assert && !globalKeyStateTracker.keysAreDown() && assert( deferredKeyboardListenersMap.size === 0,
      'There are deferred listeners but no keys are down! There is a bug or memory leak.'
    );
  }

  /**
   * Returns the first EnglishStringToCodeMap that corresponds to the provided event.code. Null if no match is found.
   * Useful when matching an english string used by KeyboardListener to the event code from a
   * SceneryEvent.domEvent.code.
   *
   * For example:
   *
   *   KeyboardUtils.eventCodeToEnglishString( 'KeyA' ) === 'a'
   *   KeyboardUtils.eventCodeToEnglishString( 'Numpad0' ) === '0'
   *   KeyboardUtils.eventCodeToEnglishString( 'Digit0' ) === '0'
   *
   * NOTE: This cannot be in KeyboardUtils because it would create a circular dependency.
   */
  public static eventCodeToEnglishString( eventCode: string ): EnglishKey | null {
    for ( const key in EnglishStringToCodeMap ) {
      if ( EnglishStringToCodeMap.hasOwnProperty( key ) &&
           ( EnglishStringToCodeMap[ key as EnglishKey ] ).includes( eventCode ) ) {
        return key as EnglishKey;
      }
    }
    return null;
  }
}

scenery.register( 'KeyboardListener', KeyboardListener );
export default KeyboardListener;
