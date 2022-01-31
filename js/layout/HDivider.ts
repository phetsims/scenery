// Copyright 2021, University of Colorado Boulder

/**
 * A vertical line for separating items in a horizontal layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Divider, HeightSizable, DividerOptions } from '../imports.js';

type HDividerOptions = DividerOptions;

class HDivider extends HeightSizable( Divider ) {
  constructor( options: HDividerOptions ) {
    super( options );

    this.preferredHeightProperty.link( height => {
      if ( height !== null ) {
        this.y2 = height;
      }
    } );
  }
}

scenery.register( 'HDivider', HDivider );
export default HDivider;
export type { HDividerOptions };
