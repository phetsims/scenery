// Copyright 2021, University of Colorado Boulder

/**
 * A vertical line for separating items in a horizontal layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Divider, HeightSizable } from '../imports.js';

class HDivider extends HeightSizable( Divider ) {
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