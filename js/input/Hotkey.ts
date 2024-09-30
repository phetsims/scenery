// Copyright 2024, University of Colorado Boulder

/**
 * Represents a single hotkey (keyboard shortcut) that can be either:
 *
 * 1. Added to globalHotkeyRegistry (to be available regardless of keyboard focus)
 * 2. Added to a node's inputListeners (to be available only when that node is part of the focused trail)
 *
 * For example:
 *
 *    globalHotkeyRegistry.add( new Hotkey( {
 *      keyStringProperty: new Property( 'y' ),
 *      fire: () => console.log( 'fire: y' )
 *    } ) );
 *
 *    myNode.addInputListener( {
 *      hotkeys: [
 *        new Hotkey( {
 *          keyStringProperty: new Property( 'x' ),
 *          fire: () => console.log( 'fire: x' )
 *        } )
 *      ]
 *    } );
 *
 * Also supports modifier keys that must be pressed in addition to the Key. See options for a description of how
 * they behave.
 *
 * Hotkeys are managed by hotkeyManager, which determines which hotkeys are active based on the globalHotkeyRegistry
 * and what Node has focus. See that class for information about how hotkeys work.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import CallbackTimer from '../../../axon/js/CallbackTimer.js';
import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import EnabledComponent, { EnabledComponentOptions } from '../../../axon/js/EnabledComponent.js';
import TProperty from '../../../axon/js/TProperty.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import optionize from '../../../phet-core/js/optionize.js';
import { AllowedKeysString, EnglishStringToCodeMap, hotkeyManager, KeyDescriptor, OneKeyStroke, scenery } from '../imports.js';

export type HotkeyFireOnHoldTiming = 'browser' | 'custom';

type SelfOptions = {

  // Describes the keys, modifier keys, and ignored modifier keys for this hotkey. This is a Property to support
  // dynamic behavior. This will be useful for i18n or creating new keymaps. See KeyDescriptor for documentation
  // about the key and modifierKeys.
  keyStringProperty: TReadOnlyProperty<OneKeyStroke>;

  // Called as fire() when the hotkey is fired (see fireOnDown/fireOnHold for when that happens).
  // The event will be null if the hotkey was fired due to fire-on-hold.
  fire?: ( event: KeyboardEvent | null ) => void;

  // Called as press() when the hotkey is pressed. Note that the Hotkey may be pressed before firing depending
  // on fireOnDown. And press is not called with fire-on-hold. The event may be null if there is a press due to
  // the hotkey becoming active due to change in state without a key press.
  press?: ( event: KeyboardEvent | null ) => void;

  // Called as release() when the Hotkey is released. Note that the Hotkey may release without calling fire() depending
  // on fireOnDown. Event may be null in cases of interrupt or if the hotkey is released due to change in state without
  // a key release.
  release?: ( event: KeyboardEvent | null ) => void;

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

  // For each main `key`, the hotkey system will only allow one hotkey with allowOverlap:false to be active at any time.
  // This is provided to allow multiple hotkeys with the same keys to fire. Default is false.
  allowOverlap?: boolean;

  // If true, any overlapping hotkeys (either added to an ancestor's inputListener or later in the local/global order)
  // will be ignored.
  override?: boolean;
};

export type HotkeyOptions = SelfOptions & EnabledComponentOptions;

export default class Hotkey extends EnabledComponent {

  // Straight from options
  public readonly keyStringProperty: TReadOnlyProperty<OneKeyStroke>;
  public readonly fire: ( event: KeyboardEvent | null ) => void;
  public readonly press: ( event: KeyboardEvent | null ) => void;
  public readonly release: ( event: KeyboardEvent | null ) => void;
  public readonly fireOnDown: boolean;
  public readonly fireOnHold: boolean;
  public readonly fireOnHoldTiming: HotkeyFireOnHoldTiming;
  public readonly allowOverlap: boolean;
  public readonly override: boolean;

  // A Property for the KeyDescriptor describing the key and modifier keys for this hotkey from the keyStringProperty.
  public readonly keyDescriptorProperty: TReadOnlyProperty<KeyDescriptor>;

  // All keys that are part of this hotkey (key + modifierKeys) as defined by the current KeyDescriptor.
  public keysProperty: TReadOnlyProperty<AllowedKeysString[]>;

  // A Property that tracks whether the hotkey is currently pressed.
  // Will be true if it meets the following conditions:
  //
  // 1. Main `key` pressed
  // 2. All modifier keys in the hotkey's `modifierKeys` are pressed
  // 3. All modifier keys not in the hotkey's `modifierKeys` (but in the other enabled hotkeys) are not pressed
  public readonly isPressedProperty: TProperty<boolean> = new BooleanProperty( false );

  // (read-only for client code)
  // Whether the last release (value during isPressedProperty => false) was due to an interruption (e.g. focus changed).
  // If false, the hotkey was released due to the key being released.
  public interrupted = false;

  // Internal timer for when fireOnHold:true and fireOnHoldTiming:custom.
  private fireOnHoldTimer?: CallbackTimer;

  public constructor(
    providedOptions: HotkeyOptions
  ) {

    assert && assert( providedOptions.fireOnHoldTiming === 'custom' || ( providedOptions.fireOnHoldCustomDelay === undefined && providedOptions.fireOnHoldCustomInterval === undefined ),
      'Cannot specify fireOnHoldCustomDelay / fireOnHoldCustomInterval if fireOnHoldTiming is not custom' );

    const options = optionize<HotkeyOptions, SelfOptions, EnabledComponentOptions>()( {
      fire: _.noop,
      press: _.noop,
      release: _.noop,
      fireOnDown: true,
      fireOnHold: false,
      fireOnHoldTiming: 'browser',
      fireOnHoldCustomDelay: 400,
      fireOnHoldCustomInterval: 100,
      allowOverlap: false,
      override: false
    }, providedOptions );

    super( options );

    // Store public things
    this.keyStringProperty = options.keyStringProperty;
    this.fire = options.fire;
    this.press = options.press;
    this.release = options.release;
    this.fireOnDown = options.fireOnDown;
    this.fireOnHold = options.fireOnHold;
    this.fireOnHoldTiming = options.fireOnHoldTiming;
    this.allowOverlap = options.allowOverlap;
    this.override = options.override;

    this.keyDescriptorProperty = new DerivedProperty( [ this.keyStringProperty ], ( keyString: OneKeyStroke ) => {
      return KeyDescriptor.keyStrokeToKeyDescriptor( keyString );
    } );

    this.keysProperty = new DerivedProperty( [ this.keyDescriptorProperty ], ( keyDescriptor: KeyDescriptor ) => {
      const keys = _.uniq( [ keyDescriptor.key, ...keyDescriptor.modifierKeys ] );

      // Make sure that every key has an entry in the EnglishStringToCodeMap
      for ( const key of keys ) {
        assert && assert( EnglishStringToCodeMap[ key ], `No codes for this key exists, do you need to add it to EnglishStringToCodeMap?: ${key}` );
      }

      return keys;
    } );

    // Create a timer to handle the optional fire-on-hold feature.
    if ( this.fireOnHold && this.fireOnHoldTiming === 'custom' ) {
      this.fireOnHoldTimer = new CallbackTimer( {
        callback: this.fire.bind( this, null ), // Pass null for fire-on-hold events
        delay: options.fireOnHoldCustomDelay,
        interval: options.fireOnHoldCustomInterval
      } );
      this.disposeEmitter.addListener( () => this.fireOnHoldTimer!.dispose() );

      this.isPressedProperty.link( isPressed => {
        // We need to reset the timer, so we stop it (even if we are starting it in just a bit again)
        this.fireOnHoldTimer!.stop( false );

        if ( isPressed ) {
          this.fireOnHoldTimer!.start();
        }
      } );
    }
  }

  /**
   * On "press" of a Hotkey. All keys are pressed while the Hotkey is active. May also fire depending on
   * events. See hotkeyManager.
   *
   * (scenery-internal)
   */
  public onPress( event: KeyboardEvent | null, shouldFire: boolean ): void {

    // clear the flag on every press (set before notifying the isPressedProperty)
    this.interrupted = false;

    this.isPressedProperty.value = true;

    // press after setting up state
    this.press( event );

    if ( shouldFire ) {
      this.fire( event );
    }
  }

  /**
   * On "release" of a Hotkey. All keys are released while the Hotkey is inactive. May also fire depending on
   * events. See hotkeyManager.
   */
  public onRelease( event: KeyboardEvent | null, interrupted: boolean, shouldFire: boolean ): void {
    this.interrupted = interrupted;

    this.isPressedProperty.value = false;

    this.release( event );

    if ( shouldFire ) {
      this.fire( event );
    }
  }

  /**
   * Manually interrupt this hotkey, releasing it and setting a flag so that it will not fire until the next time
   * keys are pressed.
   */
  public interrupt(): void {
    if ( this.isPressedProperty.value ) {
      hotkeyManager.interruptHotkey( this );
    }
  }

  public override dispose(): void {
    this.isPressedProperty.dispose();
    this.keysProperty.dispose();
    this.keyDescriptorProperty.dispose();

    super.dispose();
  }
}
scenery.register( 'Hotkey', Hotkey );