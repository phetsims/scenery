// Copyright 2021-2022, University of Colorado Boulder

/**
 * Base type for a line-divider (when put in a layout container, it will be hidden if it is before/after all visible
 * components, or if it's after another a divider in the visible order).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import optionize from '../../../../phet-core/js/optionize.js';
import { Line, LineOptions, scenery } from '../../imports.js';

export type DividerOptions = LineOptions;

export default class Divider extends Line {
  constructor( providedOptions?: LineOptions ) {
    super( optionize<LineOptions, {}, LineOptions>()( {
      layoutOptions: {
        stretch: true
      },

      // Matches HSeparator/VSeparator as a default
      stroke: 'rgb(100,100,100)'
    }, providedOptions ) );
  }
}

scenery.register( 'Divider', Divider );