// Copyright 2021-2025, University of Colorado Boulder

/**
 * Base type for a line-divider (when put in a layout container, it will be hidden if it is before/after all visible
 * components, or if it's after another a divider in the visible order).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { EmptySelfOptions, optionize3 } from '../../../../phet-core/js/optionize.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import Line from '../../nodes/Line.js';
import type { LineOptions } from '../../nodes/Line.js';
import scenery from '../../scenery.js';

type SelfOptions = EmptySelfOptions;

// Separators are automatically shown/hidden and hence should not be instrumented for PhET-iO control.
export type SeparatorOptions = SelfOptions & StrictOmit<LineOptions, 'tandem'>;

export const DEFAULT_SEPARATOR_LAYOUT_OPTIONS = {
  stretch: true,
  isSeparator: true
};

export default class Separator extends Line {
  public constructor( providedOptions?: SeparatorOptions ) {
    super( optionize3<SeparatorOptions, SelfOptions, LineOptions>()( {}, {
      layoutOptions: DEFAULT_SEPARATOR_LAYOUT_OPTIONS,

      // dark gray
      stroke: 'rgb(100,100,100)'
    }, providedOptions ) );
  }
}

scenery.register( 'Separator', Separator );