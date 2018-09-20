// Copyright 2013-2018, University of Colorado Boulder

/**
 * HBox is a convenience specialization of LayoutBox with horizontal orientation.
 *
 * @author Sam Reid
 */
define( function( require ) {
  'use strict';

  // modules
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
  function HBox( options ) {

    options = options || {};

    assert && assert( Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    assert && assert( !options.orientation, 'HBox sets orientation' );
    options.orientation = 'horizontal';

    LayoutBox.call( this, options );
  }

  scenery.register( 'HBox', HBox );

  return inherit( LayoutBox, HBox );
} );
