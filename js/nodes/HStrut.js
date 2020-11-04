// Copyright 2015-2020, University of Colorado Boulder

/**
 * A Node meant to just take up horizontal space (usually for layout purposes).
 * It is never displayed, and cannot have children.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import Spacer from './Spacer.js';

class HStrut extends Spacer {
  /**
   * Creates a strut with x in the range [0,width] and y=0.
   * @public
   *
   * @param {number} width - Width of the strut
   * @param {Object} [options] - Passed to Spacer/Node
   */
  constructor( width, options ) {
    super( width, 0, options );
  }
}

scenery.register( 'HStrut', HStrut );
export default HStrut;