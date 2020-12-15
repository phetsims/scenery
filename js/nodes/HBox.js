// Copyright 2013-2020, University of Colorado Boulder

/**
 * HBox is a convenience specialization of LayoutBox with horizontal orientation.
 *
 * @author Sam Reid
 */

import scenery from '../scenery.js';
import LayoutBox from './LayoutBox.js';

class HBox extends LayoutBox {
  /**
   * @param {Object} [options] see LayoutBox
   */
  constructor( options ) {

    options = options || {};

    assert && assert( Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    assert && assert( !options.orientation, 'HBox sets orientation' );
    options.orientation = 'horizontal';

    super( options );
  }
}

scenery.register( 'HBox', HBox );

export default HBox;