// Copyright 2023, University of Colorado Boulder

/**
 * Interface for an output raster
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector4 from '../../../../../dot/js/Vector4.js';

type OutputRaster = {
  addPartialPixel( color: Vector4, x: number, y: number ): void;
  addFullPixel( color: Vector4, x: number, y: number ): void;
  addFullRegion( color: Vector4, x: number, y: number, width: number, height: number ): void;
};

export default OutputRaster;
