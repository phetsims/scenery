// Copyright 2021-2022, University of Colorado Boulder

/**
 * A vertical line for separating items in a horizontal layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Divider, HeightSizable, DividerOptions } from '../../imports.js';

export type HDividerOptions = DividerOptions;

export default class HDivider extends HeightSizable( Divider ) {
  constructor( options?: HDividerOptions ) {
    super();

    this.localPreferredHeightProperty.link( height => {
      if ( height !== null ) {
        this.y2 = height;
      }
    } );

    this.mutate( options );
  }
}

scenery.register( 'HDivider', HDivider );