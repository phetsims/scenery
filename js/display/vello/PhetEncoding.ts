// Copyright 2023, University of Colorado Boulder

/**
 * A Vello encoding that can draw specific nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import { Affine, Color, Encoding, Extend, GradientStop, imageBitmapMap, LinearGradient, Paint, PaintDef, Pattern, RadialGradient, scenery, SourceImage, TPaint, VelloColorStop } from '../../imports.js';

const convertColor = ( color: Color ) => {
  return ( ( color.r << 24 ) + ( color.g << 16 ) + ( color.b << 8 ) + ( Math.floor( color.a * 255 ) & 0xff ) ) >>> 0;
};

const convertColorStop = ( color_stop: GradientStop ) => {
  return new VelloColorStop( color_stop.ratio, convertColor( PaintDef.toColor( color_stop.color ) ) );
};

export default class PhetEncoding extends Encoding {

  public encodePaint( paint: TPaint, baseMatrix: Matrix3 ): void {
    if ( paint instanceof Paint ) {
      if ( paint.transformMatrix && !paint.transformMatrix.isIdentity() ) {
        // We need to swap tags for these
        const encoded = this.encodeMatrix( baseMatrix.timesMatrix( paint.transformMatrix ) );
        if ( encoded ) {
          // Should pretty much always do this
          this.swapLastPathTags();
        }
      }

      if ( paint instanceof LinearGradient ) {
        // TODO: gradient transforms!
        this.encodeLinearGradient( paint.start.x, paint.start.y, paint.end.x, paint.end.y, paint.stops.map( convertColorStop ), 1, Extend.Pad );
      }
      else if ( paint instanceof RadialGradient ) {
        // TODO: gradient transforms!
        this.encodeRadialGradient( paint.start.x, paint.start.y, paint.startRadius, paint.end.x, paint.end.y, paint.endRadius, paint.stops.map( convertColorStop ), 1, Extend.Pad );
      }
      else if ( paint instanceof Pattern ) {
        const source = PhetEncoding.getSourceFromImage( paint.image, paint.image.naturalWidth, paint.image.naturalHeight );
        if ( source ) {
          this.encodeImage( new SourceImage( source.width, source.height, source ), Extend.Repeat, Extend.Repeat );
        }
        else {
          // Not yet ready, just encode a transparent color
          this.encodeColor( 0 );
        }
      }
    }
    else {
      const color = PaintDef.toColor( paint );
      this.encodeColor( convertColor( color ) );
    }
  }

  public encodeMatrix( matrix: Matrix3 ): boolean {
    return this.encodeTransform( new Affine( matrix.m00(), matrix.m10(), matrix.m01(), matrix.m11(), matrix.m02(), matrix.m12() ) );
  }

  public static getSourceFromImage( image: HTMLImageElement | HTMLCanvasElement, imageWidth = 0, imageHeight = 0 ): HTMLCanvasElement | ImageBitmap | null {
    if ( image instanceof HTMLImageElement ) {
      const imageBitmap = imageBitmapMap.get( image );
      if ( imageBitmap && imageBitmap.width && imageBitmap.height ) {
        return imageBitmap;
      }
      else if ( imageWidth && imageHeight ) {
        // We generally end up here if the imageBitmap isn't computed yet (freshly drawn something!)
        // We'll try drawing it to a Canvas

        const canvas = document.createElement( 'canvas' );
        canvas.width = imageWidth;
        canvas.height = imageHeight;

        const context = canvas.getContext( '2d' )!;
        context.drawImage( image, 0, 0 );
        return canvas;
      }
    }
    else if ( image instanceof HTMLCanvasElement ) {
      if ( image.width && image.height ) {
        return image;
      }
    }
    else {
      throw new Error( 'Unknown image type' );
    }

    return null;
  }
}

scenery.register( 'PhetEncoding', PhetEncoding );
