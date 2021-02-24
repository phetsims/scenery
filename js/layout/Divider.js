// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import Line from '../nodes/Line.js';
import scenery from '../scenery.js';

class Divider extends Line {
  /**
   * @param {Object} [options]
   */
  constructor( options ) {
    options = merge( {
      layoutOptions: {
        align: 'stretch'
      },
      stroke: 'rgb(150,150,150)'
    }, options );

    super( options );
  }
}

scenery.register( 'Divider', Divider );
export default Divider;