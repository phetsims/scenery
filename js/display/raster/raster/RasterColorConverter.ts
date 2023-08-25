// Copyright 2023, University of Colorado Boulder

/**
 * An interface that handles converting between essentially three "color spaces":
 * - client space (e.g. premultiplied sRGB)
 * - accumulation space (e.g. premultiplied linear sRGB)
 * - output space (e.g. sRGB255, so we can write to ImageData)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector4 from '../../../../../dot/js/Vector4.js';

type RasterColorConverter = {
  // NOTE: DO NOT STORE THE VALUES OF THESE RESULTS, THEY ARE MUTATED. Create a copy if needed
  clientToAccumulation( client: Vector4 ): Vector4;
  clientToOutput( client: Vector4 ): Vector4;
  accumulationToOutput( accumulation: Vector4 ): Vector4;
};

export default RasterColorConverter;
