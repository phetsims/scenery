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
   * @mixes Events
   *
   * @param {Object} [options] see LayoutBox
   */
  function VBox( options ) {
    LayoutBox.call( this, _.extend( {}, options, { orientation: 'vertical' } ) );
  }

  scenery.register( 'VBox', VBox );

  return inherit( LayoutBox, VBox );
} );
