// Copyright 2002-2014, University of Colorado Boulder

/**
 * A Node meant to just take up horizontal space (usually for layout purposes).
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
   * Creates a strut with x in the range [0,width] and y=0.
   */
  scenery.HStrut = function HStrut( width, options ) {
    Spacer.call( this, width, 0, options );
  };
  var HStrut = scenery.HStrut;

  return inherit( Spacer, HStrut );
} );
