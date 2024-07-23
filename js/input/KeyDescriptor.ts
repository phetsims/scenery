// Copyright 2024, University of Colorado Boulder

/**
 * A collection of fields that describes keys for a Hotkey. This includes a key that should be pressed to fire the
 * behavior, modifier keys that must be pressed in addition to the key, and ignored modifier keys that will not prevent
 * the hotkey from firing even if they are down.
 *
 * See the KeyDescriptorOptions for detailed information.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import { EnglishKey, OneKeyStroke, scenery } from '../imports.js';
import optionize from '../../../phet-core/js/optionize.js';

export type KeyDescriptorOptions = {

  // The key that should be pressed to trigger the hotkey (in fireOnDown:true mode) or released to trigger the hotkey
  // (in fireOnDown:false mode).
  key: EnglishKey;

  // A set of modifier keys that:
  //
  // 1. Need to be pressed before the main key before this hotkey is considered pressed.
  // 2. Must NOT be pressed for other hotkeys to be activated when this hotkey is present.
  //
  // A Hotkey will also not activate if the standard modifier keys (ctrl/alt/meta/shift) are pressed, unless they
  // are explicitly included in the modifierKeys array.
  //
  // NOTE: This is a generalization of the normal concept of "modifier key"
  // (https://en.wikipedia.org/wiki/Modifier_key). It is a PhET-specific concept that allows other non-standard
  // modifier keys to be used as modifiers. The standard modifier keys (ctrl/alt/meta/shift) are automatically handled
  // by the hotkey system, but this can expand the set of modifier keys that can be used. When a modifier key is added,
  // pressing it will prevent any other Hotkeys from becoming active. This is how the typical modifier keys behave and
  // so that is kept consistent for PhET-specific modifier keys.
  //
  // Note that the release of a modifier key may "activate" the hotkey for "fire-on-hold", but not for "fire-on-down".
  modifierKeys?: EnglishKey[];

  // A set of modifier keys that can be down and the hotkey will still fire. Essentially ignoring the modifier
  // key behavior for this key.
  ignoredModifierKeys?: EnglishKey[];
};

export default class KeyDescriptor {
  public readonly key: EnglishKey;
  public readonly modifierKeys: EnglishKey[];
  public readonly ignoredModifierKeys: EnglishKey[];

  public constructor( providedOptions?: KeyDescriptorOptions ) {
    const options = optionize<KeyDescriptorOptions>()( {
      modifierKeys: [],
      ignoredModifierKeys: []
    }, providedOptions );

    this.key = options.key;
    this.modifierKeys = options.modifierKeys;
    this.ignoredModifierKeys = options.ignoredModifierKeys;
  }

  /**
   * Returns an array of all keys that are part of this hotkey. This includes the key and all modifier keys.
   */
  public getKeysArray(): EnglishKey[] {
    return [ this.key, ...this.modifierKeys ];
  }

  /**
   * Returns a string representation of the hotkey in the format of "natural" english. Modifier keys first, followed
   * by the final key. For example, if the key is 't' and the modifier keys are 'shift', the string would be 'shift+t'.
   */
  public getHotkeyString(): OneKeyStroke[number] {
    return [
      ...this.modifierKeys,
      this.key
    ].join( '+' );
  }
}

scenery.register( 'KeyDescriptor', KeyDescriptor );