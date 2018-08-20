// Copyright 2013-2016, University of Colorado Boulder

/**
 * VBox is a convenience specialization of LayoutBox with vertical orientation.
 *
 * @author Sam Reid
 */
define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var LayoutBox = require( 'SCENERY/nodes/LayoutBox' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @public
   * @constructor
   * @extends LayoutBox
   *
   * @param {Object} [options] see LayoutBox
   */
  function VBox( options ) {
    assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    options = options || {};
    assert && assert( !options.orientation, 'VBox sets orientation' );
    options.orientation = 'vertical';

    LayoutBox.call( this, options );
  }

  scenery.register( 'VBox', VBox );

  return inherit( LayoutBox, VBox );
} );
