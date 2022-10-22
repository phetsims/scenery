// Copyright 2021-2022, University of Colorado Boulder

/**
 * Base type for a line-divider (when put in a layout container, it will be hidden if it is before/after all visible
 * components, or if it's after another a divider in the visible order).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import optionize, { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import { Line, LineOptions, scenery } from '../../imports.js';

// Separators are automatically shown/hidden and hence should not be instrumented for PhET-iO control.
export type SeparatorOptions = StrictOmit<LineOptions, 'tandem'>;

export default class Separator extends Line {
  public constructor( providedOptions?: LineOptions ) {
    super( optionize<LineOptions, EmptySelfOptions, LineOptions>()( {
      layoutOptions: {
        stretch: true,
        isSeparator: true
      },

      // Matches HSeparatorDeprecated/VSeparatorDeprecated as a default
      stroke: 'rgb(100,100,100)'
    }, providedOptions ) );
  }
}

scenery.register( 'Separator', Separator );
