// Copyright 2023, University of Colorado Boulder

/**
 * Represents queryable raster data (e.g. an image)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Color, RenderColorSpace } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

type RenderImageable = {
  width: number;
  height: number;
  colorSpace: RenderColorSpace;

  // TODO: sampling of things, actually have methods that get samples (in any color space)
  evaluate: ( point: Vector2 ) => Color;
};

export default RenderImageable;
