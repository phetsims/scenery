// Copyright 2002-2014, University of Colorado Boulder

/**
 * HBox is a convenience specialization of LayoutBox with horizontal orientation.
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
  function HBox( options ) {
    LayoutBox.call( this, _.extend( {}, options, { orientation: 'horizontal' } ) );
  }

  scenery.HBox = HBox;

  return inherit( LayoutBox, HBox );
} );