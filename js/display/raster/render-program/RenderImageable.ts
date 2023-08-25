// Copyright 2023, University of Colorado Boulder

/**
 * Represents queryable raster data (e.g. an image)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector4 from '../../../../../dot/js/Vector4.js';

type RenderImageable = {
  width: number;
  height: number;
  isFullyOpaque: boolean;

  // TODO: sampling of things, actually have methods that get samples (in any color space)
  evaluate: ( x: number, y: number ) => Vector4;
};

export default RenderImageable;

export type SerializedRenderImageable = {
  width: number;
  height: number;
  isFullyOpaque: boolean;
  data: number[];
};
