// Copyright 2013-2019, University of Colorado Boulder

/**
 * HBox is a convenience specialization of LayoutBox with horizontal orientation.
 *
 * @author Sam Reid
 */

import inherit from '../../../phet-core/js/inherit.js';
import scenery from '../scenery.js';
import LayoutBox from './LayoutBox.js';

/**
 * @public
 * @constructor
 * @extends LayoutBox
 *
 * @param {Object} [options] see LayoutBox
 */
function HBox( options ) {

  options = options || {};

  assert && assert( Object.getPrototypeOf( options ) === Object.prototype,
    'Extra prototype on Node options object is a code smell' );

  assert && assert( !options.orientation, 'HBox sets orientation' );
  options.orientation = 'horizontal';

  LayoutBox.call( this, options );
}

scenery.register( 'HBox', HBox );

inherit( LayoutBox, HBox );
export default HBox;