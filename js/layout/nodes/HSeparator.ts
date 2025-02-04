// Copyright 2021-2025, University of Colorado Boulder

/**
 * A horizontal line for separating items in a vertical layout container.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import scenery from '../../scenery.js';
import Separator from '../../layout/nodes/Separator.js';
import type { SeparatorOptions } from '../../layout/nodes/Separator.js';
import WidthSizable from '../../layout/WidthSizable.js';
import type { WidthSizableOptions } from '../../layout/WidthSizable.js';

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