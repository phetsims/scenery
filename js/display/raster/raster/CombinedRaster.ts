// Copyright 2023, University of Colorado Boulder

/**
 * An OutputRaster that tries to efficiently write straight to ImageData when possible, and otherwise accumulates.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { OutputRaster, RasterColorConverter, RasterConvertPremultipliedSRGBToSRGB255, Rasterize, scenery } from '../../../imports.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

const defaultColorConverter = new RasterConvertPremultipliedSRGBToSRGB255();

// TODO: consider implementing a raster that JUST uses ImageData, and does NOT do linear (proper) blending
export default class CombinedRaster implements OutputRaster {
  public readonly accumulationArray: Float64Array;
  public readonly imageData: ImageData;
  private combined = false;

  public constructor(
    public readonly width: number,
    public readonly height: number,
    public readonly imageDataColorspace: 'srgb' | 'display-p3' = 'srgb',
    public readonly colorConverter: RasterColorConverter = defaultColorConverter
  ) {
    this.accumulationArray = new Float64Array( width * height * 4 );

    // TODO: can we get 16 bits for display-p3?
    // TODO: can we get HDR support somehow?

    // NOTE: Firefox does not currently support display-p3
    this.imageData = imageDataColorspace === 'srgb' ?
                     new ImageData( this.width, this.height ) :
                     new ImageData( this.width, this.height, { colorSpace: imageDataColorspace } );

    assert && assert( imageDataColorspace === 'srgb' || this.imageData.colorSpace === imageDataColorspace,
      'Check with RenderColor.canvasSupportsDisplayP3() before using display-p3' );
  }

  public addClientPartialPixel( color: Vector4, x: number, y: number ): void {
    assert && assert( color.isFinite() );
    assert && assert( isFinite( x ) && isFinite( y ) );
    assert && assert( x >= 0 && x < this.width );
    assert && assert( y >= 0 && y < this.height );

    const accumulation = this.colorConverter.clientToAccumulation( color );

    const baseIndex = 4 * ( y * this.width + x );
    this.accumulationArray[ baseIndex ] += accumulation.x;
    this.accumulationArray[ baseIndex + 1 ] += accumulation.y;
    this.accumulationArray[ baseIndex + 2 ] += accumulation.z;
    this.accumulationArray[ baseIndex + 3 ] += accumulation.w;
  }

  public addClientFullPixel( color: Vector4, x: number, y: number ): void {
    assert && assert( color.isFinite() );

    this.addOutputFullPixel( this.colorConverter.clientToOutput( color ), x, y );
  }

  public addOutputFullPixel( color: Vector4, x: number, y: number ): void {
    assert && assert( color.isFinite() );
    assert && assert( isFinite( x ) && isFinite( y ) );
    assert && assert( x >= 0 && x < this.width );
    assert && assert( y >= 0 && y < this.height );

    const baseIndex = 4 * ( y * this.width + x );
    const data = this.imageData.data;
    data[ baseIndex ] = color.x;
    data[ baseIndex + 1 ] = color.y;
    data[ baseIndex + 2 ] = color.z;
    data[ baseIndex + 3 ] = color.w;
  }

  public addClientFullRegion( color: Vector4, x: number, y: number, width: number, height: number ): void {
    assert && assert( color.isFinite() );

    this.addOutputFullRegion( this.colorConverter.clientToOutput( color ), x, y, width, height );
  }

  public addOutputFullRegion( color: Vector4, x: number, y: number, width: number, height: number ): void {
    assert && assert( color.isFinite() );
    assert && assert( isFinite( x ) && isFinite( y ) );
    assert && assert( x >= 0 && x < this.width );
    assert && assert( y >= 0 && y < this.height );
    assert && assert( isFinite( width ) && isFinite( height ) );
    assert && assert( width > 0 && height > 0 );
    assert && assert( x + width <= this.width );
    assert && assert( y + height <= this.height );

    const data = this.imageData.data;

    for ( let j = 0; j < height; j++ ) {
      const rowIndex = ( y + j ) * this.width + x;
      for ( let i = 0; i < width; i++ ) {
        const baseIndex = 4 * ( rowIndex + i );
        // For debugging, useful to quickly see regions
        // data[ baseIndex ] = ( i === 0 || i === width - 1 || j === 0 || j === height - 1 ) ? 0 : sRGB.x;
        // data[ baseIndex + 1 ] = ( i === 0 || i === width - 1 || j === 0 || j === height - 1 ) ? 0 : sRGB.y;
        // data[ baseIndex + 2 ] = ( i === 0 || i === width - 1 || j === 0 || j === height - 1 ) ? 0 : sRGB.z;
        data[ baseIndex ] = color.x;
        data[ baseIndex + 1 ] = color.y;
        data[ baseIndex + 2 ] = color.z;
        data[ baseIndex + 3 ] = color.w;
      }
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

  public toCanvas(): HTMLCanvasElement {
    return Rasterize.imageDataToCanvas( this.toImageData() );
  }
}

scenery.register( 'CombinedRaster', CombinedRaster );
