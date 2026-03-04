// Copyright 2026, University of Colorado Boulder

/**
 * BooleanProperty used for Node.focusedProperty with optional metadata about what caused focus to be gained.
 *
 * The boolean value remains the primary API (`true` when focused, `false` otherwise). `focusOrigin` is additional
 * context that is set when focus is gained and cleared when focus is lost.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import type { FocusOrigin } from './SceneryEvent.js';
import scenery from '../scenery.js';

export default class FocusedBooleanProperty extends BooleanProperty {

  // Metadata describing what caused the current focused=true state. Null when unfocused or unknown.
  public focusOrigin: FocusOrigin | null = null;

  /**
   * Set focus state and accompanying focus origin metadata.
   *
   * @param focused - Whether the associated Node is focused.
   * @param focusOrigin - Optional focus source metadata for focused=true transitions.
   */
  public setFocused( focused: boolean, focusOrigin: FocusOrigin | null = null ): void {
    this.focusOrigin = focused ? focusOrigin : null;
    this.set( focused );
  }

  /**
   * Override to keep `focusOrigin` and the boolean value consistent, even when callers bypass setFocused()
   * and write directly through set()/value assignment. Any false transition clears origin metadata so stale
   * focus source information cannot leak after focus is lost.
   */
  public override set( value: boolean ): void {
    if ( !value ) {
      this.focusOrigin = null;
    }
    super.set( value );
  }
}

scenery.register( 'FocusedBooleanProperty', FocusedBooleanProperty );
