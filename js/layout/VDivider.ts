// Copyright 2021-2022, University of Colorado Boulder

/**
 * A horizontal line for separating items in a vertical layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Divider, WidthSizable, DividerOptions } from '../imports.js';

export type VDividerOptions = DividerOptions;

export default class VDivider extends WidthSizable( Divider ) {
  constructor( options?: VDividerOptions ) {
    super( options );

    this.localPreferredWidthProperty.link( width => {
      if ( width !== null ) {
        this.x2 = width;
      }
    } );
  }
}

scenery.register( 'VDivider', VDivider );
