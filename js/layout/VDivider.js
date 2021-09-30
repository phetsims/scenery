// Copyright 2021, University of Colorado Boulder

/**
 * A horizontal line for separating items in a vertical layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import Divider from './Divider.js';
import WidthSizable from './WidthSizable.js';

class VDivider extends WidthSizable( Divider ) {
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