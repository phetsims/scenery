// Copyright 2023, University of Colorado Boulder

/**
 * A simple OutputRaster that dumps everything in the accumulationBuffer, then finalizes with color space conversions
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { OutputRaster, scenery } from '../../../imports.js';
import Vector4 from '../../../../../dot/js/Vector4.js';

// TODO: type of raster that applies itself to a rectangle in the future?
export default class AccumulationRaster implements OutputRaster {
  public readonly accumulationBuffer: Vector4[] = [];

  public constructor( public readonly width: number, public readonly height: number ) {
    for ( let i = 0; i < width * height; i++ ) {
      this.accumulationBuffer.push( Vector4.ZERO.copy() );
    }
  }

  public addPartialPixel( color: Vector4, x: number, y: number ): void {
    const index = y * this.width + x;
    this.accumulationBuffer[ index ].add( color );
  }

  public addFullPixel( color: Vector4, x: number, y: number ): void {
    const index = y * this.width + x;
    this.accumulationBuffer[ index ].set( color );
  }

  public addFullRegion( color: Vector4, x: number, y: number, width: number, height: number ): void {
    for ( let j = 0; j < height; j++ ) {
      const rowIndex = ( y + j ) * this.width + x;
      for ( let i = 0; i < width; i++ ) {
        this.accumulationBuffer[ rowIndex + i ].set( color );
      }
    }
  }

  public toImageData(): ImageData {
    const imageData = new ImageData( this.width, this.height, { colorSpace: 'srgb' } );

    for ( let i = 0; i < this.accumulationBuffer.length; i++ ) {
      const accumulation = this.accumulationBuffer[ i ];
      const a = accumulation.w;

      // unpremultiply
      if ( a > 0 ) {
        const x = accumulation.x / a;
        const y = accumulation.y / a;
        const z = accumulation.z / a;

        // linear to sRGB
        const r = x <= 0.00313066844250063 ? x * 12.92 : 1.055 * Math.pow( x, 1 / 2.4 ) - 0.055;
        const g = y <= 0.00313066844250063 ? y * 12.92 : 1.055 * Math.pow( y, 1 / 2.4 ) - 0.055;
        const b = z <= 0.00313066844250063 ? z * 12.92 : 1.055 * Math.pow( z, 1 / 2.4 ) - 0.055;

        const index = 4 * i;
        imageData.data[ index ] = r * 255;
        imageData.data[ index + 1 ] = g * 255;
        imageData.data[ index + 2 ] = b * 255;
        imageData.data[ index + 3 ] = a * 255;
      }
    }

    return imageData;
  }
}

scenery.register( 'AccumulationRaster', AccumulationRaster );
