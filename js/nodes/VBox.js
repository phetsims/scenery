// Copyright 2002-2014, University of Colorado Boulder

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
   * @param {Object} [options] see LayoutBox
   * @constructor
   */
  function VBox( options ) {
    LayoutBox.call( this, _.extend( {}, options, { orientation: 'vertical' } ) );
  }

  scenery.VBox = VBox;

  return inherit( LayoutBox, VBox );
} );