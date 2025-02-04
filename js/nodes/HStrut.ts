// Copyright 2015-2025, University of Colorado Boulder

/**
 * A Node meant to just take up horizontal space (usually for layout purposes).
 * It is never displayed, and cannot have children.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import type { SpacerOptions } from '../nodes/Spacer.js';
import Spacer from '../nodes/Spacer.js';

export type HStrutOptions = SpacerOptions;

export default class HStrut extends Spacer {
  /**
   * Creates a strut with x in the range [0,width] and y=0.
   *
   * @param width - Width of the strut
   * @param [options] - Passed to Spacer/Node
   */
  public constructor( width: number, options?: HStrutOptions ) {
    super( width, 0, options );
  }
}

scenery.register( 'HStrut', HStrut );