// Copyright 2024-2025, University of Colorado Boulder

/**
 * Data pertaining to a hotkey, including keystrokes and associated metadata for documentation and the keyboard help
 * dialog.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import InstanceRegistry from '../../../phet-core/js/documentation/InstanceRegistry.js';
import optionize from '../../../phet-core/js/optionize.js';
import type { OneKeyStroke } from './KeyDescriptor.js';
import KeyDescriptor from './KeyDescriptor.js';
import scenery from '../scenery.js';

// The type for a serialized HotkeyData object for documentation (binder).
type SerializedHotkeyData = {
  keyStrings: string[];
  repoName: string;
  binderName: string;
  global: boolean;
};

export type HotkeyDataOptions = {

  // The list of keystrokes that will trigger the hotkey. Wrapping in a Property allows for i18n in the future.
  keyStringProperties: TReadOnlyProperty<OneKeyStroke>[];

  // The visual label for this Hotkey in the Keyboard Help dialog. This will also be used as the label in
  // generated documentation, unless binderName is provided.
  keyboardHelpDialogLabelStringProperty?: TReadOnlyProperty<string> | null;

  // The PDOM label and description for this Hotkey in the Keyboard Help dialog.
  keyboardHelpDialogPDOMLabelStringProperty?: TReadOnlyProperty<string> | string | null;

  // Data for binder (generated documentation).
  repoName: string; // Name of the repository where the hotkey is defined.
  global?: boolean; // Is this Hotkey global?
  binderName?: string; // If there is no keyboardHelpDialogLabelStringProperty, this name will be used in documentation.
};

export default class HotkeyData {
  public readonly keyStringProperties: TReadOnlyProperty<OneKeyStroke>[];
  public readonly keyboardHelpDialogLabelStringProperty: TReadOnlyProperty<string> | null;
  public readonly keyboardHelpDialogPDOMLabelStringProperty: TReadOnlyProperty<string> | string | null;

  // KeyDescriptors derived from keyStringProperties.
  public readonly keyDescriptorsProperty: TReadOnlyProperty<KeyDescriptor[]>;

  private readonly repoName: string;
  private readonly global: boolean;
  private readonly binderName: string;

  public constructor( providedOptions: HotkeyDataOptions ) {
    assert && assert( providedOptions.binderName || providedOptions.keyboardHelpDialogLabelStringProperty,
      'You must provide some label for the hotkey' );

    const options = optionize<HotkeyDataOptions>()( {
      keyboardHelpDialogPDOMLabelStringProperty: null,
      keyboardHelpDialogLabelStringProperty: null,
      global: false,
      binderName: ''
    }, providedOptions );

    this.keyStringProperties = options.keyStringProperties;
    this.keyboardHelpDialogLabelStringProperty = options.keyboardHelpDialogLabelStringProperty;
    this.keyboardHelpDialogPDOMLabelStringProperty = options.keyboardHelpDialogPDOMLabelStringProperty;

    this.repoName = options.repoName;
    this.global = options.global;
    this.binderName = options.binderName;

    this.keyDescriptorsProperty = DerivedProperty.deriveAny( this.keyStringProperties, () => {
      return this.keyStringProperties.map( keyStringProperty => {
        return KeyDescriptor.keyStrokeToKeyDescriptor( keyStringProperty.value );
      } );
    } );

    // Add this Hotkey to the binder registry for documentation. See documentation in the binder repository
    // for more information about how this is done.
    assert && window.phet?.chipper?.queryParameters?.binder && InstanceRegistry.registerHotkey( this );
  }

  /**
   * Returns true if any of the keyStringProperties of this HotkeyData have the given keyStroke.
   */
  public hasKeyStroke( keyStroke: OneKeyStroke ): boolean {
    return this.keyStringProperties.some( keyStringProperty => keyStringProperty.value === keyStroke );
  }

  /**
   * Serialization for usage with binder (generated documentation).
   */
  public serialize(): SerializedHotkeyData {
    return {
      keyStrings: this.keyStringProperties.map( keyStringProperty => keyStringProperty.value ),
      binderName: ( this.binderName || this.keyboardHelpDialogLabelStringProperty?.value )!,
      repoName: this.repoName,
      global: this.global
    };
  }

  /**
   * Dispose of owned Properties to prevent memory leaks.
   */
  public dispose(): void {
    this.keyDescriptorsProperty.dispose();
  }

  /**
   * Combine the keyStringProperties of an array of HotkeyData into a single array. Useful if you want to combine
   * multiple HotkeyData for a single KeyboardListener.
   */
  public static combineKeyStringProperties( hotkeyDataArray: HotkeyData[] ): TReadOnlyProperty<OneKeyStroke>[] {
    return hotkeyDataArray.reduce<TReadOnlyProperty<OneKeyStroke>[]>( ( accumulator, hotkeyData ) => {
      return accumulator.concat( hotkeyData.keyStringProperties );
    }, [] );
  }

  /**
   * Returns true if any of the HotkeyData in the array have the given keyStroke.
   */
  public static anyHaveKeyStroke( hotkeyDataArray: HotkeyData[], keyStroke: OneKeyStroke ): boolean {
    return hotkeyDataArray.some( hotkeyData => hotkeyData.hasKeyStroke( keyStroke ) );
  }
}

scenery.register( 'HotkeyData', HotkeyData );