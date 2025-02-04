// Copyright 2021-2025, University of Colorado Boulder

/**
 * A vertical line for separating items in a horizontal layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import HeightSizable from '../../layout/HeightSizable.js';
import type { HeightSizableOptions } from '../../layout/HeightSizable.js';
import scenery from '../../scenery.js';
import Separator from '../../layout/nodes/Separator.js';
import type { SeparatorOptions } from '../../layout/nodes/Separator.js';

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