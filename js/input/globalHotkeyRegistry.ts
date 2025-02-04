// Copyright 2024-2025, University of Colorado Boulder

/**
 * Stores a record of all global hotkeys (Hotkey instances that should be available regardless of focus).
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import TProperty from '../../../axon/js/TProperty.js';
import type Hotkey from '../input/Hotkey.js';
import scenery from '../scenery.js';

class GlobalHotkeyRegistry {

  // (read-only) The set of hotkeys that are currently available globally
  public readonly hotkeysProperty: TProperty<Set<Hotkey>> = new TinyProperty( new Set<Hotkey>() );

  public add( hotkey: Hotkey ): void {
    assert && assert( !this.hotkeysProperty.value.has( hotkey ), 'Hotkey already added' );

    this.hotkeysProperty.value = new Set( [ ...this.hotkeysProperty.value, hotkey ] );
  }

  public remove( hotkey: Hotkey ): void {
    assert && assert( this.hotkeysProperty.value.has( hotkey ), 'Hotkey not found' );

    this.hotkeysProperty.value = new Set( [ ...this.hotkeysProperty.value ].filter( value => value !== hotkey ) );
  }
}

scenery.register( 'GlobalHotkeyRegistry', GlobalHotkeyRegistry );

const globalHotkeyRegistry = new GlobalHotkeyRegistry();

export default globalHotkeyRegistry;