// Copyright 2021-2022, University of Colorado Boulder

/**
 * A vertical line for separating items in a horizontal layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import { Separator, SeparatorOptions, HeightSizable, HeightSizableOptions, scenery } from '../../imports.js';

type SelfOptions = EmptySelfOptions;
type ParentOptions = HeightSizableOptions & SeparatorOptions;
export type VSeparatorOptions = SelfOptions & ParentOptions;

export default class VSeparator extends HeightSizable( Separator ) {
  public constructor( options?: VSeparatorOptions ) {
    super();

    this.localPreferredHeightProperty.link( height => {
      if ( height !== null ) {
        this.y2 = height;
      }
    } );

    this.mutate( options );
  }
}

scenery.register( 'VSeparator', VSeparator );
