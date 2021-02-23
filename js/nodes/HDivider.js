// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import VSizable from '../util/VSizable.js';
import Divider from './Divider.js';

class HDivider extends VSizable( Divider ) {
  /**
   * @param {Object} [options]
   */
  constructor( options ) {
    super( options );

    this.preferredHeightProperty.link( height => {
      this.y2 = height;
    } );
  }
}

scenery.register( 'HDivider', HDivider );
export default HDivider;