// Copyright 2013-2019, University of Colorado Boulder

/**
 * A node which always fills the entire screen, no matter what the transform is.
 * Used for showing an overlay on the screen e.g., when a popup dialog is shown.
 * This can fade the background to focus on the dialog/popup as well as intercept mouse events for dismissing the dialog/popup.
 * Note: This is currently implemented using large numbers, it should be rewritten to work in any coordinate frame, possibly using kite.Shape.plane()
 * TODO: Implement using infinite geometry
 *
 * @author Sam Reid
 */

define( require => {
  'use strict';

  const inherit = require( 'PHET_CORE/inherit' );
  const scenery = require( 'SCENERY/scenery' );

  const Rectangle = require( 'SCENERY/nodes/Rectangle' );

  /**
   * @public
   * @constructor
   * @extends Rectangle
   *
   * @param {Object} [options] Passed to Rectangle. See Rectangle for more documentation
   */
  function Plane( options ) {
    Rectangle.call( this, -2000, -2000, 6000, 6000, options );
  }

  scenery.register( 'Plane', Plane );

  return inherit( Rectangle, Plane );
} );
