// Copyright 2002-2013, University of Colorado Boulder

/**
 * A node which always fills the entire screen, no matter what the transform is.
 * Used for showing an overlay on the screen e.g., when a popup dialog is shown.
 * This can fade the background to focus on the dialog/popup as well as intercept mouse events for dismissing the dialog/popup.
 * Note: This is currently implemented using large numbers, it should be rewritten to work in any coordinate frame, possibly using kite.Shape.plane()
 * TODO: Implement using infinite geometry
 *
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Shape = require( 'KITE/Shape' );

  /**
   * @param options Any options for the plane, typically fill/opacity (many others such as scale, translation, rotation should have no effect)
   * @constructor
   */
  scenery.Plane = function Plane( options ) {
    Rectangle.call( this, -1000, -1000, 3000, 3000, options );
  };
  var Plane = scenery.Plane;

  return inherit( Rectangle, Plane );
} );