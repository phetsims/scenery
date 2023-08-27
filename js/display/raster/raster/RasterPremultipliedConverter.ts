// Copyright 2023, University of Colorado Boulder

/**
 * A RasterColorConverter which handles either sRGB or Display P3 for everything but we have:
 * - client space: premultiplied
 * - accumulation space: premultiplied linear
 * - output space: 0-255 (non-premultiplied non-linear), so we can write to ImageData
 *
 * This works well to share code, since the only difference is the gamut mapping (sRGB and Display P3 have the same
 * transfer curve), and we want the same 0-255 non-premultiplied output.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector4 from '../../../../../dot/js/Vector4.js';
import { RasterColorConverter, RenderColor, scenery } from '../../../imports.js';

const scratchVector = Vector4.ZERO.copy();

export default class RasterPremultipliedConverter implements RasterColorConverter {

  protected constructor(
    public readonly gamutMap: ( color: Vector4 ) => Vector4
  ) {}

  public static readonly SRGB = new RasterPremultipliedConverter( RenderColor.gamutMapPremultipliedSRGB );
  public static readonly DISPLAY_P3 = new RasterPremultipliedConverter( RenderColor.gamutMapPremultipliedDisplayP3 );

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
    const r = client.x;
    const g = client.y;
    const b = client.z;
    const a = client.w;

    if ( a <= 0 ) {
      return scratchVector.setXYZW( 0, 0, 0, 0 );
    }
    else if ( r >= 0 && r <= 1 && g >= 0 && g <= 1 && b >= 0 && b <= 1 && a <= 1 ) {
      return scratchVector.setXYZW(
        r / a * 255,
        g / a * 255,
        b / a * 255,
        a * 255
      );
    }
    else {
      return this.gamutMap( client ).timesScalar( 255 );
    }
  }

  // NOTE: DO NOT STORE THE VALUES OF THESE RESULTS, THEY ARE MUTATED. Create a copy if needed
  public accumulationToOutput( accumulation: Vector4 ): Vector4 {

    // premultiplied linear sRGB => sRGB255, so we'll unpremultiply, convert linear => sRGB, then scale

    const a = accumulation.w;

    if ( a <= 0 ) {
      return scratchVector.setXYZW( 0, 0, 0, 0 );
    }
    else {
      // unpremultiply
      const x = accumulation.x / a;
      const y = accumulation.y / a;
      const z = accumulation.z / a;

      // linear to sRGB
      const r = x <= 0.00313066844250063 ? x * 12.92 : 1.055 * Math.pow( x, 1 / 2.4 ) - 0.055;
      const g = y <= 0.00313066844250063 ? y * 12.92 : 1.055 * Math.pow( y, 1 / 2.4 ) - 0.055;
      const b = z <= 0.00313066844250063 ? z * 12.92 : 1.055 * Math.pow( z, 1 / 2.4 ) - 0.055;

      if ( r >= 0 && r <= 1 && g >= 0 && g <= 1 && b >= 0 && b <= 1 && a <= 1 ) {
        return scratchVector.setXYZW(
          r * 255,
          g * 255,
          b * 255,
          a * 255
        );
      }
      else {
        return this.gamutMap( scratchVector.setXYZW( r * a, g * a, b * a, a ) ).timesScalar( 255 );
      }
    }
  }
}

scenery.register( 'RasterPremultipliedConverter', RasterPremultipliedConverter );
