// Copyright 2023, University of Colorado Boulder

/**
 * A RasterColorConverter with:
 * - client space: premultiplied sRGB
 * - accumulation space: premultiplied linear sRGB
 * - output space: sRGB255, so we can write to ImageData
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector4 from '../../../../../dot/js/Vector4.js';
import { RasterColorConverter, scenery } from '../../../imports.js';

const scratchVector = Vector4.ZERO.copy();

export default class RasterConvertPremultipliedSRGBToSRGB255 implements RasterColorConverter {
  // NOTE: DO NOT STORE THE VALUES OF THESE RESULTS, THEY ARE MUTATED. Create a copy if needed
  public clientToAccumulation( client: Vector4 ): Vector4 {

    // premultiplied sRGB => premultiplied linear sRGB, so we'll have to unpremultiply, sRGB => linear sRGB, then premultiply

    const a = client.w;

    if ( a > 0 ) {
      // unpremultiply
      const x = client.x / a;
      const y = client.y / a;
      const z = client.z / a;

      // sRGB => linear sRGB, WITH the premultiply
      return scratchVector.setXYZW(
        a * ( x <= 0.0404482362771082 ? x / 12.92 : Math.pow( ( x + 0.055 ) / 1.055, 2.4 ) ),
        a * ( y <= 0.0404482362771082 ? y / 12.92 : Math.pow( ( y + 0.055 ) / 1.055, 2.4 ) ),
        a * ( z <= 0.0404482362771082 ? z / 12.92 : Math.pow( ( z + 0.055 ) / 1.055, 2.4 ) ),
        a
      );
    }
    else {
      return scratchVector.setXYZW( 0, 0, 0, 0 );
    }

  }

  // NOTE: DO NOT STORE THE VALUES OF THESE RESULTS, THEY ARE MUTATED. Create a copy if needed
  public clientToOutput( client: Vector4 ): Vector4 {

    // premultiplied sRGB => sRGB255, so we'll unpremultiply and then scale

    const a = client.w;

    if ( a > 0 ) {
      return scratchVector.setXYZW(
        client.x / a * 255,
        client.y / a * 255,
        client.z / a * 255,
        a * 255
      );
    }
    else {
      return scratchVector.setXYZW( 0, 0, 0, 0 );
    }
  }

  // NOTE: DO NOT STORE THE VALUES OF THESE RESULTS, THEY ARE MUTATED. Create a copy if needed
  public accumulationToOutput( accumulation: Vector4 ): Vector4 {

    // premultiplied linear sRGB => sRGB255, so we'll unpremultiply, convert linear => sRGB, then scale

    const a = accumulation.w;

    if ( a > 0 ) {
      // unpremultiply
      const x = accumulation.x / a;
      const y = accumulation.y / a;
      const z = accumulation.z / a;

      // linear to sRGB
      const r = x <= 0.00313066844250063 ? x * 12.92 : 1.055 * Math.pow( x, 1 / 2.4 ) - 0.055;
      const g = y <= 0.00313066844250063 ? y * 12.92 : 1.055 * Math.pow( y, 1 / 2.4 ) - 0.055;
      const b = z <= 0.00313066844250063 ? z * 12.92 : 1.055 * Math.pow( z, 1 / 2.4 ) - 0.055;

      return scratchVector.setXYZW(
        r * 255,
        g * 255,
        b * 255,
        a * 255
      );
    }
    else {
      return scratchVector.setXYZW( 0, 0, 0, 0 );
    }
  }
}

scenery.register( 'RasterConvertPremultipliedSRGBToSRGB255', RasterConvertPremultipliedSRGBToSRGB255 );
