// Copyright 2013-2020, University of Colorado Boulder

/**
 * A node which always fills the entire screen, no matter what the transform is.
 * Used for showing an overlay on the screen e.g., when a popup dialog is shown.
 * This can fade the background to focus on the dialog/popup as well as intercept mouse events for dismissing the dialog/popup.
 * Note: This is currently implemented using large numbers, it should be rewritten to work in any coordinate frame, possibly using kite.Shape.plane()
 * TODO: Implement using infinite geometry
 *
 * @author Sam Reid
 */

import scenery from '../scenery.js';
import Rectangle from './Rectangle.js';

class Plane extends Rectangle {
  /**
   * @param {Object} [options] Passed to Rectangle. See Rectangle for more documentation
   */
  constructor( options ) {
    super( -2000, -2000, 6000, 6000, options );
  }
}

scenery.register( 'Plane', Plane );

export default Plane;