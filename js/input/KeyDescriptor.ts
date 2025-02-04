// Copyright 2024-2025, University of Colorado Boulder

/**
 * A collection of fields that describes keys for a Hotkey. This includes a key that should be pressed to fire the
 * behavior, modifier keys that must be pressed in addition to the key, and ignored modifier keys that will not prevent
 * the hotkey from firing even if they are down.
 *
 * See the KeyDescriptorOptions for detailed information.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import optionize from '../../../phet-core/js/optionize.js';
import type { EnglishKey, EnglishKeyString } from '../accessibility/EnglishStringToCodeMap.js';
import EnglishStringToCodeMap, { metaEnglishKeys } from '../accessibility/EnglishStringToCodeMap.js';
import scenery from '../scenery.js';

// NOTE: The typing for ModifierKey and OneKeyStroke is limited TypeScript, there is a limitation to the number of
//       entries in a union type. If that limitation is not acceptable remove this typing. OR maybe TypeScript will
//       someday support regex patterns for a type. See https://github.com/microsoft/TypeScript/issues/41160
// If we run out of union space for template strings, consider the above comment or remove some from the type.
type ModifierKey = 'q' | 'w' | 'e' | 'r' | 't' | 'y' | 'u' | 'i' | 'o' | 'p' | 'a' | 's' | 'd' |
  'f' | 'g' | 'h' | 'j' | 'k' | 'l' | 'z' | 'x' | 'c' |
  'v' | 'b' | 'n' | 'm' | 'ctrl' | 'alt' | 'shift' | 'tab' | 'meta';

const IGNORE_DELIMITER = '?';
type IgnoreDelimiter = typeof IGNORE_DELIMITER;

type IgnoreModifierKey = `${ModifierKey}${IgnoreDelimiter}`;
type IgnoreOtherModifierKeys = `${IgnoreDelimiter}${ModifierKey}`;

// Allowed keys are the keys of the EnglishStringToCodeMap.
type AllowedKeys = keyof typeof EnglishStringToCodeMap;

// Allowed keys as a string - the format they will be provided by the user.
export type AllowedKeysString = `${AllowedKeys}`;

// A key stroke entry is a single key or a key with "ignore" modifiers, see examples and keyStrokeToKeyDescriptor.
export type OneKeyStrokeEntry = `${AllowedKeys}` | `${IgnoreModifierKey}+${EnglishKey}` | `${IgnoreOtherModifierKeys}+${EnglishKey}`;

export type OneKeyStroke =
  `${AllowedKeys}` | // e.g. 't'
  `${ModifierKey}+${AllowedKeys}` | // e.g. 'shift+t'
  `${ModifierKey}+${ModifierKey}+${AllowedKeys}` | // e.g. 'ctrl+shift+t'
  `${IgnoreModifierKey}+${AllowedKeys}` | // e.g. 'shift?+t' (shift added to ignoredModifierKeys)
  `${IgnoreModifierKey}+${ModifierKey}+${AllowedKeys}` | // e.g. 'shift?+ctrl+t' (shift added to ignoredModifierKeys)
  `${IgnoreOtherModifierKeys}+${AllowedKeys}` | // e.g. '?shift+t' (shift is a modifier key but ALL other default modifier keys are ignored)
  `${IgnoreOtherModifierKeys}+${ModifierKey}+${AllowedKeys}`; // e.g. '?shift+j+t' (shift is a modifier key but ALL other default modifier keys are ignored)
// These combinations are not supported by TypeScript: "TS2590: Expression produces a union type that is too complex to
// represent." See above note and https://github.com/microsoft/TypeScript/issues/41160#issuecomment-1287271132.
// `${AllowedKeys}+${AllowedKeys}+${AllowedKeys}+${AllowedKeys}`;
// type KeyCombinations = `${OneKeyStroke}` | `${OneKeyStroke},${OneKeyStroke}`;

export type KeyDescriptorOptions = {

  // The key that should be pressed to trigger the hotkey (in fireOnDown:true mode) or released to trigger the hotkey
  // (in fireOnDown:false mode).
  key: AllowedKeysString;

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
  modifierKeys?: AllowedKeysString[];

  // A set of modifier keys that can be down and the hotkey will still fire. Essentially ignoring the modifier
  // key behavior for this key.
  ignoredModifierKeys?: AllowedKeysString[];
};

export default class KeyDescriptor {
  public readonly key: AllowedKeysString;
  public readonly modifierKeys: AllowedKeysString[];
  public readonly ignoredModifierKeys: AllowedKeysString[];

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
   * Returns a string representation of the hotkey in the format of "natural" english. Modifier keys first, followed
   * by the final key. For example, if the key is 't' and the modifier keys are 'shift', the string would be 'shift+t'.
   */
  public getHotkeyString(): OneKeyStroke[number] {
    return [
      ...this.modifierKeys,
      this.key
    ].join( '+' );
  }

  /**
   * Parses an input string to extract the main key and its associated modifier keys, while considering ignored
   * modifier keys based on the placement of the '?' delimiter.
   *
   * See KeyboardListener for a description of key and modifierKey behavior.
   *
   * The function handles the following cases:
   * 1. If a word is followed by '?', it is added to `ignoredModifierKeys`.
   * 2. If a word is preceded by '?', it indicates all other default modifier keys should be ignored,
   *    except the word itself, which is added to `modifierKeys`.
   *
   * keyStrokeToKeyDescriptor('r');
   * // Output: { key: 'r', modifierKeys: [], ignoredModifierKeys: [] }
   *
   * keyStrokeToKeyDescriptor('alt+r');
   * // Output: { key: 'r', modifierKeys: ['alt'], ignoredModifierKeys: [] }
   *
   * keyStrokeToKeyDescriptor('alt+j+r');
   * // Output: { key: 'r', modifierKeys: ['alt', 'j'], ignoredModifierKeys: [] }
   *
   * keyStrokeToKeyDescriptor('alt?+j+r');
   * // Output: { key: 'r', modifierKeys: ['j'], ignoredModifierKeys: ['alt'] }
   *
   * keyStrokeToKeyDescriptor('shift?+t');
   * // Output: { key: 't', modifierKeys: [], ignoredModifierKeys: ['shift'] }
   *
   * keyStrokeToKeyDescriptor('?shift+t');
   * // Output: { key: 't', modifierKeys: ['shift'], ignoredModifierKeys: ['alt', 'control', 'meta'] }
   *
   * keyStrokeToKeyDescriptor('?shift+t+j');
   * // Output: { key: 'j', modifierKeys: ['shift', 't'], ignoredModifierKeys: ['alt', 'control', 'meta'] }
   *
   */
  public static keyStrokeToKeyDescriptor( keyStroke: OneKeyStroke ): KeyDescriptor {

    const tokens = keyStroke.split( '+' ) as OneKeyStrokeEntry[];

    // assertions
    let foundIgnoreDelimiter = false;
    tokens.forEach( token => {

      // the ignore delimiter can only be used on default modifier keys
      if ( token.length > 1 && token.includes( IGNORE_DELIMITER ) ) {
        assert && assert( !foundIgnoreDelimiter, 'There can only be one ignore delimiter' );
        assert && assert( metaEnglishKeys.includes( token.replace( IGNORE_DELIMITER, '' ) as EnglishKeyString ), 'The ignore delimiter can only be used on default modifier keys' );
        foundIgnoreDelimiter = true;
      }
    } );

    const modifierKeys: AllowedKeysString[] = [];
    const ignoredModifierKeys: AllowedKeysString[] = [];

    tokens.forEach( token => {

      // Check if the token contains a question mark
      if ( token.includes( IGNORE_DELIMITER ) ) {
        const strippedToken = token.replace( IGNORE_DELIMITER, '' );

        if ( token.startsWith( IGNORE_DELIMITER ) ) {

          // Add all default modifiers except the current stripped token to the ignored keys
          const otherModifiers = metaEnglishKeys.filter( mod => mod !== strippedToken );
          ignoredModifierKeys.push( ...otherModifiers );

          // Include the stripped token as a regular modifier key
          modifierKeys.push( strippedToken as AllowedKeysString );
        }
        else {

          // Add the stripped token to the ignored modifier keys
          ignoredModifierKeys.push( strippedToken as AllowedKeysString );
        }
      }
      else {

        // If there's no question mark, add the token to the modifier keys
        modifierKeys.push( token as AllowedKeysString );
      }
    } );

    // Assume the last token is the key
    const key = modifierKeys.pop()!;

    // Filter out ignored modifier keys from the modifier keys list
    const filteredModifierKeys = modifierKeys.filter( mod => !ignoredModifierKeys.includes( mod ) );

    return new KeyDescriptor( {
      key: key,
      modifierKeys: filteredModifierKeys,
      ignoredModifierKeys: ignoredModifierKeys
    } );
  }
}

scenery.register( 'KeyDescriptor', KeyDescriptor );