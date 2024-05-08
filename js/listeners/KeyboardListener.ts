// Copyright 2022-2024, University of Colorado Boulder

/**
 * A listener for general keyboard input. Specify the keys with a `keys` option in a readable format that looks like
 * this: [ 'shift+t', 'alt+shift+r' ]
 *
 * - Each entry in the array represents a combination of keys that must be pressed to fire a callback.
 * - '+' separates each key in a single combination.
 * - The keys leading up to the last key in the combination are considered "modifier" keys. The last key in the
 *   combination needs to be pressed while the modifier keys are down to fire the callback.
 * - The order modifier keys are pressed does not matter for firing the callback.
 *
 * In the above example "shift+t" OR "alt+shift+r" will fire the callback when pressed.
 *
 * An example usage would like this:
 *
 * this.addInputListener( new KeyboardListener( {
 *   keys: [ 'a+b', 'a+c', 'shift+arrowLeft', 'alt+g+t', 'ctrl+3', 'alt+ctrl+t' ],
 *   fire: ( event, keysPressed, listener ) => {
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
 * By default, the fire callback will fire when the last key is pressed down. See additional options for firing on key
 * release or other press and hold behavior.
 *
 * **Important Modifier Key Behavior**
 * Modifier keys prevent other key combinations from firing their behavior if they are pressed down.
 * For example, if you have a listener with 'shift+t' and 'y', pressing 'shift' will prevent 'y' from firing.
 * This behavior matches the behavior of the browser and is intended to prevent unexpected behavior. However,
 * this behavior is also true for custom (non-standard) modifier keys. For example, if you have a listener with
 * 'j+t' and 'y', pressing 'j' will prevent 'y' from firing. This is a PhET specific design decision, but it
 * is consistent with typical modifier key behavior.
 *
 * **Global Keyboard Listeners**
 * A KeyboardListener can be added to a Node with addInputListener, and it will fire with normal scenery input dispatch
 * behavior when the Node has focus. However, a KeyboardListener can also be added "globally", meaning it will fire
 * regardless of where focus is in the document. Use KeyboardListener.createGlobal. This adds Hotkeys to the
 * globalHotkeyRegistry. Be sure to dispose of the KeyboardListener when it is no longer needed.
 *
 * **Potential Pitfall!**
 * The fire callback is only called if exactly the keys in a group are pressed. If you need to listen to a modifier key,
 * you must include it in the keys array. For example if you add a listener for 'tab', you must ALSO include
 * 'shift+tab' in the array to observe 'shift+tab' presses. If you provide 'tab' alone, the callback will not fire
 * if 'shift' is also pressed.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import optionize, { EmptySelfOptions } from '../../../phet-core/js/optionize.js';
import { DisplayedTrailsProperty, EnglishKey, EnglishStringToCodeMap, EventContext, globalHotkeyRegistry, Hotkey, HotkeyFireOnHoldTiming, Node, PDOMPointer, scenery, SceneryEvent, TInputListener, Trail } from '../imports.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import EnabledComponent, { EnabledComponentOptions } from '../../../axon/js/EnabledComponent.js';
import Property from '../../../axon/js/Property.js';
import TProperty from '../../../axon/js/TProperty.js';

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

type KeyboardListenerSelfOptions<Keys extends readonly OneKeyStroke[ ]> = {

  // The keys that need to be pressed to fire the callback. In a form like `[ 'shift+t', 'alt+shift+r' ]`. See top
  // level documentation for more information and an example of providing keys.
  keys: Keys;

  // Called when the listener detects that the set of keys are pressed.
  fire?: ( event: KeyboardEvent | null, keysPressed: Keys[number], listener: KeyboardListener<Keys> ) => void;

  // Called when the listener detects that the set of keys are pressed. Press is always called on the first press of
  // keys, but does not continue with fire-on-hold behavior. Will be called before fire if fireOnDown is true.
  press?: ( event: KeyboardEvent | null, keysPressed: Keys[number], listener: KeyboardListener<Keys> ) => void;

  // Called when the listener detects that the set of keys have been released. keysPressed may be null
  // in cases of interruption.
  release?: ( event: KeyboardEvent | null, keysPressed: Keys[number] | null, listener: KeyboardListener<Keys> ) => void;

  // Called when the listener target receives focus.
  focus?: ( listener: KeyboardListener<Keys> ) => void;

  // Called when the listener target loses focus.
  blur?: ( listener: KeyboardListener<Keys> ) => void;

  // If true, the hotkey will fire when the hotkey is initially pressed.
  // If false, the hotkey will fire when the hotkey is finally released.
  fireOnDown?: boolean;

  // Whether the fire-on-hold feature is enabled
  fireOnHold?: boolean;

  // Whether we should listen to the browser's fire-on-hold timing, or use our own.
  fireOnHoldTiming?: HotkeyFireOnHoldTiming;

  // Start to fire continuously after pressing for this long (milliseconds)
  fireOnHoldCustomDelay?: number;

  // Fire continuously at this interval (milliseconds)
  fireOnHoldCustomInterval?: number;

  // Controls whether the keys used by this KeyboardListener are allowed to overlap with other KeyboardListeners
  // that are listening for the same keys. If true, the KeyboardListener will fire even if another KeyboardListener.
  // This is implemented with Hotkey, see Hotkey.ts for more information.
  allowOverlap?: boolean;

  // If true, Keyboard listeners with overlapping keys (either added to an ancestor's inputListener or later in the
  // local/global order) will be ignored. Only the most 'local' Hotkey will fire. The default is true for
  // KeyboardListeners added to focusable Nodes, and false for global KeyboardListeners to catch overlapping global
  // keys.
  override?: boolean;

  // If you want the callback to fire when a particular modifier key is down, you can add that key to this list.
  // For example, you may want to fire the callback even when 'shift' is down, but not when another modifier key
  // is pressed.
  ignoredModifierKeys?: EnglishKey[];
};

export type KeyboardListenerOptions<Keys extends readonly OneKeyStroke[]> = KeyboardListenerSelfOptions<Keys> & EnabledComponentOptions;

class KeyboardListener<Keys extends readonly OneKeyStroke[]> extends EnabledComponent implements TInputListener {

  // from options
  private readonly _fire: ( event: KeyboardEvent | null, keysPressed: Keys[number], listener: KeyboardListener<Keys> ) => void;
  private readonly _press: ( event: KeyboardEvent | null, keysPressed: Keys[number], listener: KeyboardListener<Keys> ) => void;
  private readonly _release: ( event: KeyboardEvent | null, keysPressed: Keys[number] | null, listener: KeyboardListener<Keys> ) => void;
  private readonly _focus: ( listener: KeyboardListener<Keys> ) => void;
  private readonly _blur: ( listener: KeyboardListener<Keys> ) => void;
  public readonly fireOnDown: boolean;
  public readonly fireOnHold: boolean;
  public readonly fireOnHoldTiming: HotkeyFireOnHoldTiming;
  public readonly fireOnHoldCustomDelay: number;
  public readonly fireOnHoldCustomInterval: number;
  public readonly allowOverlap: boolean;
  public readonly ignoredModifierKeys: EnglishKey[];
  private readonly override: boolean;

  public readonly hotkeys: Hotkey[];

  // A Property that is true when any of the keys
  public readonly isPressedProperty: TReadOnlyProperty<boolean>;

  // A Property that has the list of currenty pressed keys, from the keys array.
  public readonly pressedKeysProperty: TProperty<Array<Keys[number]>>;

  // (read-only) - Whether the last key press was interrupted. Will be valid until the next press.
  public interrupted: boolean;

  public constructor( providedOptions: KeyboardListenerOptions<Keys> ) {
    const options = optionize<KeyboardListenerOptions<Keys>, KeyboardListenerSelfOptions<Keys>, EnabledComponentOptions>()( {
      fire: _.noop,
      press: _.noop,
      release: _.noop,
      focus: _.noop,
      blur: _.noop,
      fireOnDown: true,
      fireOnHold: false,
      fireOnHoldTiming: 'browser',
      fireOnHoldCustomDelay: 400,
      fireOnHoldCustomInterval: 100,
      allowOverlap: false,
      override: true,
      ignoredModifierKeys: [],

      // EnabledComponent
      // By default, do not instrument the enabledProperty; opt in with this option. See EnabledComponent
      phetioEnabledPropertyInstrumented: false
    }, providedOptions );

    super( options );

    this._fire = options.fire;
    this._press = options.press;
    this._release = options.release;
    this._focus = options.focus;
    this._blur = options.blur;
    this.fireOnDown = options.fireOnDown;
    this.fireOnHold = options.fireOnHold;
    this.fireOnHoldTiming = options.fireOnHoldTiming;
    this.fireOnHoldCustomDelay = options.fireOnHoldCustomDelay;
    this.fireOnHoldCustomInterval = options.fireOnHoldCustomInterval;
    this.allowOverlap = options.allowOverlap;
    this.ignoredModifierKeys = options.ignoredModifierKeys;
    this.override = options.override;

    // convert the provided keys to data that we can respond to with scenery's Input system
    this.hotkeys = this.createHotkeys( options.keys );

    this.isPressedProperty = DerivedProperty.or( this.hotkeys.map( hotkey => hotkey.isPressedProperty ) );
    this.pressedKeysProperty = new Property( [] );
    this.interrupted = false;

    this.enabledProperty.lazyLink( this.onEnabledPropertyChange.bind( this ) );
  }

  /**
   * Whether this listener is currently activated with a press.
   */
  public get isPressed(): boolean {
    return this.isPressedProperty.value;
  }

  /**
   * Fired when the enabledProperty changes
   */
  private onEnabledPropertyChange( enabled: boolean ): void {
    !enabled && this.interrupt();
  }

  /**
   * Dispose of this listener by disposing of any Callback timers. Then clear all KeyGroups.
   */
  public override dispose(): void {
    ( this as unknown as TInputListener ).hotkeys!.forEach( hotkey => hotkey.dispose() );
    this.isPressedProperty.dispose();
    this.pressedKeysProperty.dispose();
    super.dispose();
  }

  /**
   * Everything that uses a KeyboardListener should prevent more global scenery keyboard behavior, such as pan/zoom
   * from arrow keys.
   */
  public keydown( event: SceneryEvent<KeyboardEvent> ): void {
    event.pointer.reserveForKeyboardDrag();
  }

  /**
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
   * Part of the scenery listener API. On cancel, clear active KeyGroups and stop their behavior.
   */
  public cancel(): void {
    this.handleInterrupt();
  }

  /**
   * Part of the scenery listener API. Clear active KeyGroups and stop their callbacks.
   */
  public interrupt(): void {
    this.handleInterrupt();
  }

  /**
   * Work to be done on both cancel and interrupt.
   */
  private handleInterrupt(): void {

    // interrupt all hotkeys (will do nothing if hotkeys are interrupted or not active)
    this.hotkeys.forEach( hotkey => hotkey.interrupt() );
  }

  protected createSyntheticEvent( pointer: PDOMPointer ): SceneryEvent {
    const context = EventContext.createSynthetic();
    return new SceneryEvent( new Trail(), 'synthetic', pointer, context );
  }

  /**
   * Converts the provided keys for this listener into a collection of Hotkeys to easily track what keys are down.
   */
  private createHotkeys( keys: Keys ): Hotkey[] {
    return keys.map( naturalKeys => {

      // Split the keys into the main key and the modifier keys
      const keys = naturalKeys.split( '+' );
      const modifierKeys = keys.slice( 0, keys.length - 1 );
      const naturalKey = keys[ keys.length - 1 ];

      const hotkey = new Hotkey( {
        key: naturalKey as EnglishKey,
        modifierKeys: modifierKeys as EnglishKey[],
        ignoredModifierKeys: this.ignoredModifierKeys,
        fire: ( event: KeyboardEvent | null ) => {
          this._fire( event, naturalKeys, this );
        },
        press: ( event: KeyboardEvent | null ) => {
          this.interrupted = false;
          this._press( event, naturalKeys, this );

          assert && assert( !this.pressedKeysProperty.value.includes( naturalKeys ), 'Key already pressed' );
          this.pressedKeysProperty.value = [ ...this.pressedKeysProperty.value, naturalKeys ];
        },
        release: ( event: KeyboardEvent | null ) => {
          this.interrupted = hotkey.interrupted;

          this._release( event, naturalKeys, this );

          assert && assert( this.pressedKeysProperty.value.includes( naturalKeys ), 'Key not pressed' );
          this.pressedKeysProperty.value = this.pressedKeysProperty.value.filter( key => key !== naturalKeys );
        },
        fireOnDown: this.fireOnDown,
        fireOnHold: this.fireOnHold,
        fireOnHoldTiming: this.fireOnHoldTiming,
        fireOnHoldCustomDelay: this.fireOnHoldTiming === 'custom' ? this.fireOnHoldCustomDelay : undefined,
        fireOnHoldCustomInterval: this.fireOnHoldTiming === 'custom' ? this.fireOnHoldCustomInterval : undefined,
        allowOverlap: this.allowOverlap,
        enabledProperty: this.enabledProperty,
        override: this.override
      } );

      return hotkey;
    } );
  }

  /**
   * Adds a global KeyboardListener to a target Node. This listener will fire regardless of where focus is in
   * the document. The listener is returned so that it can be disposed.
   */
  public static createGlobal( target: Node, providedOptions: KeyboardListenerOptions<OneKeyStroke[]> ): KeyboardListener<OneKeyStroke[]> {
    return new GlobalKeyboardListener( target, providedOptions );
  }
}

/**
 * Inner class for a KeyboardListener that is global with a target Node. The listener will fire no matter where
 * focus is in the document, as long as the target Node can receive input events. Create this listener with
 * KeyboardListener.createGlobal.
 */
class GlobalKeyboardListener extends KeyboardListener<OneKeyStroke[]> {

  // All of the Trails to the target Node that can receive alternative input events.
  private readonly displayedTrailsProperty: DisplayedTrailsProperty;

  // Derived from above, whether the target Node is 'enabled' to receive input events.
  private readonly globallyEnabledProperty: TReadOnlyProperty<boolean>;

  public constructor( target: Node, providedOptions: KeyboardListenerOptions<OneKeyStroke[]> ) {
    const displayedTrailsProperty = new DisplayedTrailsProperty( target, {

      // For alt input events, use the PDOM trail to determine if the trail is displayed. This may be different
      // from the "visual" trail if the Node is placed in a PDOM order that is different from the visual order.
      followPDOMOrder: true,

      // Additionally, the target must have each of these true up its Trails to receive alt input events.
      requirePDOMVisible: true,
      requireEnabled: true,
      requireInputEnabled: true
    } );
    const globallyEnabledProperty = new DerivedProperty( [ displayedTrailsProperty ], ( trails: Trail[] ) => {
      return trails.length > 0;
    } );

    const options = optionize<KeyboardListenerOptions<OneKeyStroke[]>, EmptySelfOptions, KeyboardListenerOptions<OneKeyStroke[]>>()( {

      // The enabledProperty is forwarded to the Hotkeys so that they are disabled when the target cannot receive input.
      enabledProperty: globallyEnabledProperty
    }, providedOptions );

    super( options );

    this.displayedTrailsProperty = displayedTrailsProperty;
    this.globallyEnabledProperty = globallyEnabledProperty;

    // Add all global keys to the registry
    this.hotkeys.forEach( hotkey => {
      globalHotkeyRegistry.add( hotkey );
    } );
  }

  /**
   * Dispose Properties and remove all Hotkeys from the global registry.
   */
  public override dispose(): void {

    // Remove all global keys from the registry.
    this.hotkeys.forEach( hotkey => {
      globalHotkeyRegistry.remove( hotkey );
    } );

    this.globallyEnabledProperty.dispose();
    this.displayedTrailsProperty.dispose();
    super.dispose();
  }
}

scenery.register( 'KeyboardListener', KeyboardListener );
export default KeyboardListener;