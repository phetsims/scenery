// Copyright 2015-2021, University of Colorado Boulder

/**
 * A Node meant to just take up vertical space (usually for layout purposes).
 * It is never displayed, and cannot have children.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Spacer } from '../imports.js';

class VStrut extends Spacer {
  /**
   * Creates a strut with x=0 and y in the range [0,height].
   * @public
   *
   * @param {number} height - Height of the strut
   * @param {Object} [options] - Passed to Spacer/Node
   */
  constructor( height, options ) {
    super( 0, height, options );
  }
}

scenery.register( 'VStrut', VStrut );
export default VStrut;