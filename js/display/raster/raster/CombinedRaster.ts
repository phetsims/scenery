// Copyright 2023, University of Colorado Boulder

/**
 * An OutputRaster that tries to efficiently write straight to ImageData when possible, and otherwise accumulates.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { OutputRaster, scenery } from '../../../imports.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

const scratchCombinedVector = new Vector4( 0, 0, 0, 0 );

// TODO: consider implementing a raster that JUST uses ImageData, and does NOT do linear (proper) blending
export default class CombinedRaster implements OutputRaster {
  public readonly accumulationArray: Float64Array;
  public readonly imageData: ImageData;
  private combined = false;

  public constructor( public readonly width: number, public readonly height: number ) {
    this.accumulationArray = new Float64Array( width * height * 4 );
    this.imageData = new ImageData( this.width, this.height, { colorSpace: 'srgb' } );
  }

  public addPartialPixel( color: Vector4, x: number, y: number ): void {
    assert && assert( color.isFinite() );
    assert && assert( isFinite( x ) && isFinite( y ) );
    assert && assert( x >= 0 && x < this.width );
    assert && assert( y >= 0 && y < this.height );

    const baseIndex = 4 * ( y * this.width + x );
    this.accumulationArray[ baseIndex ] += color.x;
    this.accumulationArray[ baseIndex + 1 ] += color.y;
    this.accumulationArray[ baseIndex + 2 ] += color.z;
    this.accumulationArray[ baseIndex + 3 ] += color.w;
  }

  public addFullPixel( color: Vector4, x: number, y: number ): void {
    assert && assert( color.isFinite() );
    assert && assert( isFinite( x ) && isFinite( y ) );
    assert && assert( x >= 0 && x < this.width );
    assert && assert( y >= 0 && y < this.height );

    // Be lazy, we COULD convert here, but we'll just do it at the end
    this.addPartialPixel( color, x, y );
  }

  public addFullRegion( color: Vector4, x: number, y: number, width: number, height: number ): void {
    assert && assert( color.isFinite() );
    assert && assert( isFinite( x ) && isFinite( y ) );
    assert && assert( x >= 0 && x < this.width );
    assert && assert( y >= 0 && y < this.height );
    assert && assert( isFinite( width ) && isFinite( height ) );
    assert && assert( width > 0 && height > 0 );
    assert && assert( x + width <= this.width );
    assert && assert( y + height <= this.height );

    const sRGB = CombinedRaster.convertToSRGB( color );
    for ( let j = 0; j < height; j++ ) {
      const rowIndex = ( y + j ) * this.width + x;
      for ( let i = 0; i < width; i++ ) {
        const baseIndex = 4 * ( rowIndex + i );
        const data = this.imageData.data;
        data[ baseIndex ] = sRGB.x;
        data[ baseIndex + 1 ] = sRGB.y;
        data[ baseIndex + 2 ] = sRGB.z;
        data[ baseIndex + 3 ] = sRGB.w;
      }
    }
  }

  // TODO: can we combine these methods of sRGB conversion without losing performance?
  // TODO: move this somewhere?
  private static convertToSRGB( color: Vector4 ): Vector4 {
    const accumulation = color;
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

      return scratchCombinedVector.setXYZW(
        r * 255,
        g * 255,
        b * 255,
        a * 255
      );
    }
    else {
      return scratchCombinedVector.setXYZW( 0, 0, 0, 0 );
    }
  }

  public toImageData(): ImageData {
    if ( !this.combined ) {
      const quantity = this.accumulationArray.length / 4;
      for ( let i = 0; i < quantity; i++ ) {
        const baseIndex = i * 4;
        const a = this.accumulationArray[ baseIndex + 3 ];

        if ( a > 0 ) {
          // unpremultiply
          const x = this.accumulationArray[ baseIndex ] / a;
          const y = this.accumulationArray[ baseIndex + 1 ] / a;
          const z = this.accumulationArray[ baseIndex + 2 ] / a;

          // linear to sRGB
          const r = x <= 0.00313066844250063 ? x * 12.92 : 1.055 * Math.pow( x, 1 / 2.4 ) - 0.055;
          const g = y <= 0.00313066844250063 ? y * 12.92 : 1.055 * Math.pow( y, 1 / 2.4 ) - 0.055;
          const b = z <= 0.00313066844250063 ? z * 12.92 : 1.055 * Math.pow( z, 1 / 2.4 ) - 0.055;

          const index = 4 * i;
          // NOTE: ADDING HERE!!!! Don't change (we've set this for some pixels already)
          // Also, if we have a weird case where something sneaks in that is above an epsilon, so that we have a
          // barely non-zero linear value, we DO NOT want to wipe away something that saw a "almost full" pixel and
          // wrote into the imageData.
          this.imageData.data[ index ] += r * 255;
          this.imageData.data[ index + 1 ] += g * 255;
          this.imageData.data[ index + 2 ] += b * 255;
          this.imageData.data[ index + 3 ] += a * 255;
        }
      }
      this.combined = true;
    }

    return this.imageData;
  }
}

scenery.register( 'CombinedRaster', CombinedRaster );
