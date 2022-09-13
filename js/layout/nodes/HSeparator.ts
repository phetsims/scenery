// Copyright 2021-2022, University of Colorado Boulder

/**
 * A horizontal line for separating items in a vertical layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import { Separator, SeparatorOptions, scenery, WidthSizable, WidthSizableOptions } from '../../imports.js';

type SelfOptions = EmptySelfOptions;
type ParentOptions = WidthSizableOptions & SeparatorOptions;
export type HSeparatorOptions = SelfOptions & ParentOptions;

export default class HSeparator extends WidthSizable( Separator ) {
  public constructor( options?: HSeparatorOptions ) {
    super();

    this.localPreferredWidthProperty.link( width => {
      if ( width !== null ) {
        this.x2 = width;
      }
    } );

    this.mutate( options );
  }
}

scenery.register( 'HSeparator', HSeparator );
