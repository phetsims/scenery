// Copyright 2021-2022, University of Colorado Boulder

/**
 * Base type for a line-divider (when put in a layout container, it will be hidden if it is before/after all visible
 * components, or if it's after another a divider in the visible order).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import { scenery, Line, LineOptions } from '../imports.js';

type DividerOptions = LineOptions;

class Divider extends Line {
  constructor( options?: LineOptions ) {
    options = merge( {
      layoutOptions: {
        align: 'stretch'
      },
      stroke: 'rgb(100,100,100)'
    }, options );

    super( options );
  }
}

scenery.register( 'Divider', Divider );
export default Divider;
export type { DividerOptions };
