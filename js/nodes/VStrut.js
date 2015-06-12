// Copyright 2002-2014, University of Colorado Boulder

/**
 * A Node meant to just take up vertical space (usually for layout purposes).
 * It is never displayed, and cannot have children.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var Spacer = require( 'SCENERY/nodes/Spacer' );

  /**
   * Creates a strut with x=0 and y in the range [0,height]. Use x/y in options to control its position.
   */
  scenery.VStrut = function VStrut( height, options ) {
    Spacer.call( this, 0, height, options );
  };
  var VStrut = scenery.VStrut;

  return inherit( Spacer, VStrut );
} );
