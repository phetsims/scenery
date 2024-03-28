// Copyright 2024, University of Colorado Boulder

/**
 * IMPORTANT: EXPERIMENTAL!
 * TODO DO NOT USE IN SIMULATIONS, SEE https://github.com/phetsims/scenery/issues/1621 FIRST
 *
 * Represents a single hotkey (keyboard shortcut) that can be either:
 *
 * 1. Added to globalHotkeyRegistry (to be available regardless of keyboard focus)
 * 2. Added to a node's inputListeners (to be available only when that node is part of the focused trail)
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EnglishKey, scenery } from '../imports.js';
import optionize from '../../../phet-core/js/optionize.js';
import EnabledComponent, { EnabledComponentOptions } from '../../../axon/js/EnabledComponent.js';
import TProperty from '../../../axon/js/TProperty.js';
import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import CallbackTimer from '../../../axon/js/CallbackTimer.js';

export type HotkeyFireOnHoldTiming = 'browser' | 'custom';

type SelfOptions = {
  // The key that should be pressed to trigger the hotkey (in fireOnDown:true mode) or released to trigger the hotkey (in
  // fireOnDown:false mode).
  key: EnglishKey;

  // A set of modifier keys that:
  //
  // 1. Need to be pressed before the main key before this hotkey is considered pressed.
  // 2. If not a normal (ctrl/alt/meta/shift) modifier key, will also be required to be "off" for other hotkeys to be
  //    activated when this hotkey is present.
  //
  // Note that the release of a modifier key may "activate" the hotkey for "fire-on-hold", but not for "fire-on-down".
  modifierKeys?: EnglishKey[];

  // A set of modifier keys that can be down and the hotkey will still fire. Essentially ignoring the modifier
  // key behavior for this key.
  ignoredModifierKeys?: EnglishKey[];

  // Called as fire() when the hotkey is fired (see fireOnDown/fireOnHold for when that happens).
  // The event will be null if the hotkey was fired due to fire-on-hold.
  fire?: ( event: KeyboardEvent | null ) => void;

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
};

export type HotkeyOptions = SelfOptions & EnabledComponentOptions;

export default class Hotkey extends EnabledComponent {

  // Straight from options
  public readonly key: EnglishKey;
  public readonly modifierKeys: EnglishKey[];
  public readonly ignoredModifierKeys: EnglishKey[];
  public readonly fire: ( event: KeyboardEvent | null ) => void;
  public readonly fireOnDown: boolean;
  public readonly fireOnHold: boolean;
  public readonly fireOnHoldTiming: HotkeyFireOnHoldTiming;
  public readonly allowOverlap: boolean;

  // All keys that are part of this hotkey (key + modifierKeys)
  public readonly keys: EnglishKey[];

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
      modifierKeys: [],
      ignoredModifierKeys: [],
      fire: _.noop,
      fireOnDown: true,
      fireOnHold: false,
      fireOnHoldTiming: 'browser',
      fireOnHoldCustomDelay: 400,
      fireOnHoldCustomInterval: 100,
      allowOverlap: false
    }, providedOptions );

    super( options );

    // Store public things
    this.key = options.key;
    this.modifierKeys = options.modifierKeys;
    this.ignoredModifierKeys = options.ignoredModifierKeys;
    this.fire = options.fire;
    this.fireOnDown = options.fireOnDown;
    this.fireOnHold = options.fireOnHold;
    this.fireOnHoldTiming = options.fireOnHoldTiming;
    this.allowOverlap = options.allowOverlap;

    this.keys = _.uniq( [ this.key, ...this.modifierKeys ] );

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

  public getHotkeyString(): string {
    return [
      ...this.modifierKeys,
      this.key
    ].join( '+' );
  }

  public override dispose(): void {
    this.isPressedProperty.dispose();

    super.dispose();
  }
}
scenery.register( 'Hotkey', Hotkey );