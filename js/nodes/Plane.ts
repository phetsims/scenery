// Copyright 2013-2023, University of Colorado Boulder

/**
 * A node which always fills the entire screen, no matter what the transform is.
 * Used for showing an overlay on the screen e.g., when a popup dialog is shown.
 * This can fade the background to focus on the dialog/popup as well as intercept mouse events for dismissing the dialog/popup.
 * Note: This is currently implemented using large numbers, it should be rewritten to work in any coordinate frame, possibly using phet.kite.Shape.plane()
 * TODO: Implement using infinite geometry
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import { Rectangle, RectangleOptions, scenery } from '../imports.js';

export type PlaneOptions = RectangleOptions;

export default class Plane extends Rectangle {
  public constructor( options?: PlaneOptions ) {
    super( -2000, -2000, 6000, 6000, options );
  }
}

scenery.register( 'Plane', Plane );
