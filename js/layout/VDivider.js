// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import Divider from './Divider.js';
import HSizable from './HSizable.js';

class VDivider extends HSizable( Divider ) {
  /**
   * @param {Object} [options]
   */
  constructor( options ) {
    super( options );

    this.preferredWidthProperty.link( width => {
      this.x2 = width;
    } );
  }
}

scenery.register( 'VDivider', VDivider );
export default VDivider;