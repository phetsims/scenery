// Copyright 2013-2019, University of Colorado Boulder

/**
 * VBox is a convenience specialization of LayoutBox with vertical orientation.
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
function VBox( options ) {

  options = options || {};

  assert && assert( Object.getPrototypeOf( options ) === Object.prototype,
    'Extra prototype on Node options object is a code smell' );

  assert && assert( !options.orientation, 'VBox sets orientation' );
  options.orientation = 'vertical';

  LayoutBox.call( this, options );
}

scenery.register( 'VBox', VBox );

inherit( LayoutBox, VBox );
export default VBox;