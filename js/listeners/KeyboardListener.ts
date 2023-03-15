// Copyright 2022-2023, University of Colorado Boulder

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
 *   callback: ( event, listener ) => {
 *     const keysPressed = listener.keysPressed;
 *
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
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import CallbackTimer from '../../../axon/js/CallbackTimer.js';
import optionize from '../../../phet-core/js/optionize.js';
import { EnglishStringToCodeMap, FocusManager, globalKeyStateTracker, scenery, SceneryEvent, TInputListener } from '../imports.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';

// NOTE: The typing for ModifierKey and OneKeyStroke is limited TypeScript, there is a limitation to the number of
//       entries in a union type. If that limitation is not acceptable remove this typing. OR maybe TypeScript will
//       someday support regex patterns for a type. See https://github.com/microsoft/TypeScript/issues/41160
// If we run out of union space for template strings, consider the above comment or remove some from the type.
type ModifierKey = 'q' | 'w' | 'e' | 'r' | 't' | 'y' | 'u' | 'i' | 'o' | 'p' | 'a' | 's' | 'd' |
  'f' | 'g' | 'h' | 'j' | 'k' | 'l' | 'z' | 'x' | 'c' |
  'v' | 'b' | 'n' | 'm' | 'ctrl' | 'alt' | 'shift' | 'tab';
type AllowedKeys = keyof typeof EnglishStringToCodeMap;

export type OneKeyStroke = `${AllowedKeys}` |
  `${ModifierKey}+${AllowedKeys}` |
  `${ModifierKey}+${ModifierKey}+${AllowedKeys}`;
// These combinations are not supported by TypeScript: "TS2590: Expression produces a union type that is too complex to
// represent." See above note and https://github.com/microsoft/TypeScript/issues/41160#issuecomment-1287271132.
// `${AllowedKeys}+${AllowedKeys}+${AllowedKeys}+${AllowedKeys}`;
// type KeyCombinations = `${OneKeyStroke}` | `${OneKeyStroke},${OneKeyStroke}`;

// Possible input types that decide when the callbacks of this listener should fire.
// - 'up': Callbacks fire on release of keys.
// - 'down': Callbacks fire on press of keys.
// - 'both': Callbacks fire on both press and release of keys.
type ListenerFireTrigger = 'up' | 'down' | 'both';

type KeyboardListenerOptions<Keys extends readonly OneKeyStroke[ ]> = {

  // The keys that need to be pressed to fire the callback. In a form like `[ 'shift+t', 'alt+shift+r' ]`. See top
  // level documentation for more information and an example of providing keys.
  keys: Keys;

  // If true, the listener will fire callbacks if keys other than keys in the key group happen to be down at the same
  // time. If false, callbacks will fire only when the keys of a group are exclusively down. Setting this to true is
  // also useful if you want multiple key groups from your provided keys to fire callbacks at the same time.
  allowOtherKeys?: boolean;

  // If true, the listener will fire for keys regardless of where focus is in the document. Use this when you want
  // to add some key press behavior that will always fire no matter what the event target is. If this listener
  // is added to a Node, it will only fire if the Node (and all of its ancestors) are visible with inputEnabled: true.
  // More specifically, this uses `globalKeyUp` and `globalKeyDown`. See definitions in Input.ts for more information.
  global?: boolean;

  // If true, this listener is fired during the 'capture' phase, meaning BEFORE other listeners get fired during
  // typical event dispatch. Only relevant for `global` key events.
  capture?: boolean;

  // If true, all SceneryEvents that trigger this listener (keydown and keyup) will be `handled` (no more
  // event bubbling). See `manageEvent` for more information.
  handle?: boolean;

  // If true, all SceneryEvents that trigger this listener (keydown and keyup) will be `aborted` (no more
  // event bubbling, no more listeners fire). See `manageEvent` for more information.
  abort?: boolean;

  // Called when the listener detects that the set of keys are pressed.
  callback?: ( event: SceneryEvent<KeyboardEvent> | null, listener: KeyboardListener<Keys> ) => void;

  // Called when the listener is cancelled/interrupted.
  cancel?: ( listener: KeyboardListener<Keys> ) => void;

  // When true, the listener will fire continuously while keys are held down, at the following intervals.
  fireOnHold?: boolean;

  // If fireOnHold true, this is the delay in (in milliseconds) before the callback is fired continuously.
  fireOnHoldDelay?: number;

  // If fireOnHold true, this is the interval (in milliseconds) that the callback fires after the fireOnHoldDelay.
  fireOnHoldInterval?: number;

  // Possible input types that decide when callbacks of the listener fire in response to input. See
  // ListenerFireTrigger type documentation.
  listenerFireTrigger?: ListenerFireTrigger;
};

type KeyGroup<Keys extends readonly OneKeyStroke[]> = {

  // All must be pressed fully before the key is pressed to activate the command.
  modifierKeys: string[];

  // the final key that is pressed (after modifier keys) to trigger the listener
  key: string;

  // all keys in this KeyGroup as a KeyboardEvent.code
  allKeys: string[];

  // All keys in this KeyGroup using the readable form
  naturalKeys: Keys[number];

  // A callback timer for this KeyGroup to support press and hold timing and callbacks
  timer: CallbackTimer | null;
};

class KeyboardListener<Keys extends readonly OneKeyStroke[]> implements TInputListener {

  // The function called when a KeyGroup is pressed (or just released). Provides the SceneryEvent that fired the input
  // listeners and this the keys that were pressed from the active KeyGroup. The event may be null when using
  // fireOnHold or in cases of cancel or interrupt.
  private readonly _callback: ( event: SceneryEvent<KeyboardEvent> | null, listener: KeyboardListener<Keys> ) => void;

  // The optional function called when this listener is cancelled.
  private readonly _cancel: ( listener: KeyboardListener<Keys> ) => void;

  // When callbacks are fired in response to input. Could be on keys pressed down, up, or both.
  private readonly _listenerFireTrigger: ListenerFireTrigger;

  // Does the listener fire the callback continuously when keys are held down?
  private readonly _fireOnHold: boolean;

  // (scenery-internal) All the KeyGroups of this listener from the keys provided in natural language.
  public readonly _keyGroups: KeyGroup<Keys>[];

  // All the KeyGroups that are currently firing
  private readonly _activeKeyGroups: KeyGroup<Keys>[];

  // Current keys pressed that are having their listeners fired now.
  public keysPressed: Keys[number] | null = null;

  // True when keys are pressed down. If listenerFireTrigger is 'both', you can look at this in your callback to
  // determine if keys are pressed or released.
  public keysDown: boolean;

  // Timing variables for the CallbackTimers.
  private readonly _fireOnHoldDelay: number;
  private readonly _fireOnHoldInterval: number;

  // see options documentation
  private readonly _global: boolean;
  private readonly _handle: boolean;
  private readonly _abort: boolean;
  private readonly _allowOtherKeys: boolean;

  private readonly _windowFocusListener: ( windowHasFocus: boolean ) => void;

  public constructor( providedOptions: KeyboardListenerOptions<Keys> ) {
    const options = optionize<KeyboardListenerOptions<Keys>>()( {
      callback: _.noop,
      cancel: _.noop,
      global: false,
      capture: false,
      handle: false,
      abort: false,
      listenerFireTrigger: 'down',
      fireOnHold: false,
      fireOnHoldDelay: 400,
      fireOnHoldInterval: 100,
      allowOtherKeys: false
    }, providedOptions );

    this._callback = options.callback;
    this._cancel = options.cancel;

    this._listenerFireTrigger = options.listenerFireTrigger;
    this._fireOnHold = options.fireOnHold;
    this._fireOnHoldDelay = options.fireOnHoldDelay;
    this._fireOnHoldInterval = options.fireOnHoldInterval;
    this._allowOtherKeys = options.allowOtherKeys;

    this._activeKeyGroups = [];

    this.keysDown = false;

    this._global = options.global;
    this._handle = options.handle;
    this._abort = options.abort;

    // convert the provided keys to data that we can respond to with scenery's Input system
    this._keyGroups = this.convertKeysToKeyGroups( options.keys );

    ( this as unknown as TInputListener ).listener = this;
    ( this as unknown as TInputListener ).capture = options.capture;

    this._windowFocusListener = this.handleWindowFocusChange.bind( this );
    FocusManager.windowHasFocusProperty.link( this._windowFocusListener );
  }

  /**
   * Mostly required to fire with CallbackTimer since the callback cannot take arguments.
   */
  public fireCallback( event: SceneryEvent<KeyboardEvent> | null, keyGroup: KeyGroup<Keys> ): void {
    this.keysPressed = keyGroup.naturalKeys;
    this._callback( event, this );
    this.keysPressed = null;
  }

  /**
   * Part of the scenery listener API. Responding to a keydown event, update active KeyGroups and potentially
   * fire callbacks and start CallbackTimers.
   */
  private handleKeyDown( event: SceneryEvent<KeyboardEvent> ): void {
    if ( this._listenerFireTrigger === 'down' || this._listenerFireTrigger === 'both' ) {

      // modifier keys can be pressed in any order but the last key in the group must be pressed last
      this._keyGroups.forEach( keyGroup => {

        if ( !this._activeKeyGroups.includes( keyGroup ) ) {
          if ( this.areKeysDownForListener( keyGroup.allKeys ) &&
               KeyboardUtils.areKeysEquivalent( keyGroup.key, globalKeyStateTracker.getLastKeyDown()! ) ) {

            this._activeKeyGroups.push( keyGroup );

            this.keysDown = true;

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
   * fires a KeyGroup if the key that was released has all other modifier keys down.
   */
  private handleKeyUp( event: SceneryEvent<KeyboardEvent> ): void {

    if ( this._activeKeyGroups.length > 0 ) {
      this._activeKeyGroups.forEach( ( activeKeyGroup, index ) => {
        if ( !this.areKeysDownForListener( activeKeyGroup.allKeys ) ) {
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
          if ( this.areKeysDownForListener( keyGroup.modifierKeys ) &&
               KeyboardUtils.areKeysEquivalent( keyGroup.key, eventCode ) ) {
            this.keysDown = false;
            this.fireCallback( event, keyGroup );
          }
        } );
      }
    }

    this.manageEvent( event );
  }

  /**
   * Are the provided keys currently pressed in a way that should start or stop firing callbacks? If this listener
   * allows other keys to be pressed, returns true if the keys are down. If not, it returns true if ONLY the
   * provided keys are down.
   */
  private areKeysDownForListener( keys: string[] ): boolean {
    return this._allowOtherKeys ? globalKeyStateTracker.areKeysDown( keys ) : globalKeyStateTracker.areKeysExclusivelyDown( keys );
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
    if ( !this._global ) {
      this.handleKeyDown( event );
    }
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the keyup event when not
   * added to the global key events. Target will be the Node, Display, or Pointer this listener was added to.
   */
  public keyup( event: SceneryEvent<KeyboardEvent> ): void {
    if ( !this._global ) {
      this.handleKeyUp( event );
    }
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the global keydown event.
   * Event has no target.
   */
  public globalkeydown( event: SceneryEvent<KeyboardEvent> ): void {
    if ( this._global ) {
      this.handleKeyDown( event );
    }
  }

  /**
   * This is part of the scenery Input API (implementing TInputListener). Handle the global keyup event.
   * Event has no target.
   */
  public globalkeyup( event: SceneryEvent<KeyboardEvent> ): void {
    if ( this._global ) {
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
   * Dispose of this listener by disposing of any Callback timers. Then clear all KeyGroups.
   */
  public dispose(): void {
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

      const naturalKey = groupKeys.slice( -1 )[ 0 ];
      const key = EnglishStringToCodeMap[ naturalKey ];
      assert && assert( key, `Key not found, do you need to add it to EnglishStringToCodeMap? ${naturalKey}` );

      let modifierKeys: string[] = [];
      if ( groupKeys.length > 1 ) {
        modifierKeys = groupKeys.slice( 0, groupKeys.length - 1 ).map( naturalModifierKey => {
          const modifierKey = EnglishStringToCodeMap[ naturalModifierKey ];
          assert && assert( modifierKey, `Key not found, do you need to add it to EnglishStringToCodeMap? ${naturalModifierKey}` );
          return modifierKey;
        } );
      }

      // Set up the timer for triggering callbacks if this listener supports press and hold behavior
      const timer = this._fireOnHold ? new CallbackTimer( {
        callback: () => this.fireCallback( null, keyGroup ),
        delay: this._fireOnHoldDelay,
        interval: this._fireOnHoldInterval
      } ) : null;

      const keyGroup: KeyGroup<Keys> = {
        key: key,
        modifierKeys: modifierKeys,
        naturalKeys: naturalKeys,
        allKeys: modifierKeys.concat( key ),
        timer: timer
      };
      return keyGroup;
    } );

    return keyGroups;
  }
}

scenery.register( 'KeyboardListener', KeyboardListener );
export default KeyboardListener;
