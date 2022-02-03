// Copyright 2013-2021, University of Colorado Boulder

/**
 * A node which always fills the entire screen, no matter what the transform is.
 * Used for showing an overlay on the screen e.g., when a popup dialog is shown.
 * This can fade the background to focus on the dialog/popup as well as intercept mouse events for dismissing the dialog/popup.
 * Note: This is currently implemented using large numbers, it should be rewritten to work in any coordinate frame, possibly using kite.Shape.plane()
 * TODO: Implement using infinite geometry
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import { scenery, Rectangle, RectangleOptions } from '../imports.js';

type PlaneOptions = RectangleOptions;

class Plane extends Rectangle {
  constructor( options?: PlaneOptions ) {
    super( -2000, -2000, 6000, 6000, options );
  }
}

scenery.register( 'Plane', Plane );

export default Plane;
export type { PlaneOptions };
