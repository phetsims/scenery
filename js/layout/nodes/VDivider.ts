// Copyright 2021-2022, University of Colorado Boulder

/**
 * A horizontal line for separating items in a vertical layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Divider, WidthSizable, DividerOptions, WidthSizableOptions } from '../../imports.js';

type SelfOptions = {};
type ParentOptions = WidthSizableOptions & DividerOptions;
export type VDividerOptions = SelfOptions & ParentOptions;

export default class VDivider extends WidthSizable( Divider ) {
  constructor( options?: VDividerOptions ) {
    super();

    this.localPreferredWidthProperty.link( width => {
      if ( width !== null ) {
        this.x2 = width;
      }
    } );

    this.mutate( options );
  }
}

scenery.register( 'VDivider', VDivider );
