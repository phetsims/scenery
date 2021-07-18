// Copyright 2013-2020, University of Colorado Boulder

/**
 * VBox is a convenience specialization of LayoutBox with vertical orientation.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import scenery from '../scenery.js';
import LayoutBox from './LayoutBox.js';

class VBox extends LayoutBox {
  /**
   * @param {Object} [options] see LayoutBox
   */
  constructor( options ) {

    options = options || {};

    assert && assert( Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    assert && assert( !options.orientation, 'VBox sets orientation' );
    options.orientation = 'vertical';

    super( options );
  }
}

scenery.register( 'VBox', VBox );

export default VBox;