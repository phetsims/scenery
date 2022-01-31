// Copyright 2021, University of Colorado Boulder

/**
 * A horizontal line for separating items in a vertical layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Divider, WidthSizable, DividerOptions } from '../imports.js';

type VDividerOptions = DividerOptions;

class VDivider extends WidthSizable( Divider ) {
  constructor( options: VDividerOptions ) {
    super( options );

    this.preferredWidthProperty.link( width => {
      if ( width !== null ) {
        this.x2 = width;
      }
    } );
  }
}

scenery.register( 'VDivider', VDivider );
export default VDivider;
export type { VDividerOptions };
