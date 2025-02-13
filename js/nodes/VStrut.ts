// Copyright 2015-2025, University of Colorado Boulder

/**
 * A Node meant to just take up vertical space (usually for layout purposes).
 * It is never displayed, and cannot have children.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import type { SpacerOptions } from '../nodes/Spacer.js';
import Spacer from '../nodes/Spacer.js';

export type VStrutOptions = SpacerOptions;

export default class VStrut extends Spacer {
  /**
   * Creates a strut with x=0 and y in the range [0,height].
   *
   * @param height - Height of the strut
   * @param [options] - Passed to Spacer/Node
   */
  public constructor( height: number, options?: VStrutOptions ) {
    super( 0, height, options );
  }
}

scenery.register( 'VStrut', VStrut );