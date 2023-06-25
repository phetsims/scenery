// Copyright 2023, University of Colorado Boulder

/**
 * A Vello encoding that can draw specific nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { default as Encoding, Extend, Mix, Compose, VelloColorStop, base64ToU8 } from './Encoding.js';
import { default as wasmInit, load_font_data, shape_text, get_glyph } from './swash.js';
import Arial from './.Arial.js'; // eslint-disable-line default-import-match-filename
import swash_wasm from './swash_wasm.js';
import Matrix3 from '../../../../dot/js/Matrix3.js';
import { Shape } from '../../../../kite/js/imports.js';
import { LinearGradient, Paint, PaintDef, RadialGradient, scenery, Node, Text, Image, Color, GradientStop, TPaint, Path } from '../../imports.js';
import { Affine } from './Affine.js';
import { BufferImage } from './BufferImage.js';

let loaded = false;

// TODO: use scenery imports for things to avoid circular reference issues

const matrixToAffine = ( matrix: Matrix3 ) => new Affine( matrix.m00(), matrix.m10(), matrix.m01(), matrix.m11(), matrix.m02(), matrix.m12() );

const convert_color = ( color: Color ) => {
  return ( ( color.r << 24 ) + ( color.g << 16 ) + ( color.b << 8 ) + ( Math.floor( color.a * 255 ) & 0xff ) ) >>> 0;
};

const convert_color_stop = ( color_stop: GradientStop ) => {
  return new VelloColorStop( color_stop.ratio, convert_color( PaintDef.toColor( color_stop.color ) ) );
};

export default class PhetEncoding extends Encoding {

  public static async load(): Promise<void> {
    if ( !loaded ) {
      await wasmInit( base64ToU8( swash_wasm ) );

      load_font_data( base64ToU8( Arial ) );

      console.log( 'Fonts loaded' );
      loaded = true;
    }
  }

  public encode_paint( paint: TPaint ): void {
    if ( paint instanceof Paint ) {
      if ( paint instanceof LinearGradient ) {
        this.encode_linear_gradient( paint.start.x, paint.start.y, paint.end.x, paint.end.y, paint.stops.map( convert_color_stop ), 1, Extend.Pad );
      }
      else if ( paint instanceof RadialGradient ) {
        this.encode_radial_gradient( paint.start.x, paint.start.y, paint.startRadius, paint.end.x, paint.end.y, paint.endRadius, paint.stops.map( convert_color_stop ), 1, Extend.Pad );
      }
      else {
        // Pattern, no-op for now
        // TODO: implement pattern, shouldn't be too hard
        console.log( 'PATTERN UNIMPLEMENTED' );
        this.encode_color( 0 );
      }
    }
    else {
      const color = PaintDef.toColor( paint );
      this.encode_color( convert_color( color ) );
    }
  }

  public encode_node( topNode: Node ): void {
    const matrixStack = [ Matrix3.IDENTITY ];

    const recurse = ( node: Node ) => {
      if ( !node.visible || !node.bounds.isValid() ) {
        return;
      }

      const hasClip = node.opacity !== 1 || node.clipArea;

      let matrix = matrixStack[ matrixStack.length - 1 ];
      if ( !node.matrix.isIdentity() ) {
        matrix = matrix.timesMatrix( node.matrix );
        matrixStack.push( matrix );
      }

      if ( hasClip ) {
        this.encode_transform( matrixToAffine( matrix ) );

        this.encode_linewidth( -1 );

        if ( node.clipArea ) {
          this.encode_kite_shape( node.clipArea, true, true, 1 );
        }
        else {
          // Just handling opacity
          const safeBoundingBox = node.localBounds.dilated( 100 ); // overdo it, how to clip without shape?
          const safeBoundingShape = Shape.bounds( safeBoundingBox );

          this.encode_kite_shape( safeBoundingShape, true, true, 1 );
        }

        this.encode_begin_clip( node.opacity === 1 ? Mix.Normal : Mix.Normal, Compose.SrcOver, node.opacity );
      }

      if ( node instanceof Path ) {
        if ( node.shape ) {
          if ( node.hasFill() ) {
            this.encode_transform( matrixToAffine( matrix ) );
            this.encode_linewidth( -1 );
            this.encode_kite_shape( node.shape, true, true, 1 );
            this.encode_paint( node.fill );
          }
          if ( node.hasStroke() ) {
            this.encode_transform( matrixToAffine( matrix ) );
            let shape = node.shape;
            if ( node.lineDash.length ) {
              shape = node.shape.getDashedShape( node.lineDash, node.lineDashOffset );
            }
            this.encode_linewidth( node.lineWidth );
            this.encode_kite_shape( shape, false, true, 1 );
            this.encode_paint( node.stroke );
          }
        }
      }

      // TODO: support stroked text
      if ( node instanceof Text ) {
        if ( node.hasFill() ) {
          const shapedText: {
            id: number;
            x: number;
            y: number;
            adv: number;
          }[] = JSON.parse( shape_text( node.renderedText, true ) );

          let hasEncodedGlyph = false;

          // TODO: more performance possible easily
          const scale = node._font.numericSize / 2048; // get UPM
          const sizedMatrix = matrix.timesMatrix( Matrix3.scaling( scale ) );
          const shearMatrix = Matrix3.rowMajor(
            // approx 14 degrees, with always vertical flip
            1, node._font.style !== 'normal' ? 0.2419 : 0, 0,
            0, -1, 0, // vertical flip
            0, 0, 1
          );

          let embolden = 0;
          if ( node._font.weight === 'bold' ) {
            embolden = 40;
          }

          let x = 0;
          shapedText.forEach( glyph => {
            const shape = new Shape( get_glyph( glyph.id, embolden, embolden ) ); // TODO: bold! (italic with oblique transform!!)

            // TODO: check whether the glyph y needs to be reversed! And italics/oblique
            const glyphMatrix = sizedMatrix.timesMatrix( Matrix3.translation( x + glyph.x, glyph.y ) ).timesMatrix( shearMatrix );
            x += glyph.adv;

            this.encode_transform( new Affine(
              glyphMatrix.m00(), glyphMatrix.m10(), glyphMatrix.m01(), glyphMatrix.m11(),
              glyphMatrix.m02(), glyphMatrix.m12()
            ) );
            this.encode_linewidth( -1 );
            const encodedCount = this.encode_kite_shape( shape, true, false, 1 );
            if ( encodedCount ) {
              hasEncodedGlyph = true;
            }
          } );

          if ( hasEncodedGlyph ) {
            this.insert_path_marker();
            this.encode_paint( node.fill );
          }
        }
      }

      // TODO: not too hard to implement pattern now
      if ( node instanceof Image && node._image ) {
        const canvas = document.createElement( 'canvas' );
        canvas.width = node.getImageWidth();
        canvas.height = node.getImageHeight();

        // if we are not loaded yet, just ignore
        if ( canvas.width && canvas.height ) {
          const context = canvas.getContext( '2d' )!;
          context.drawImage( node._image, 0, 0 );
          const imageData = context.getImageData( 0, 0, canvas.width, canvas.height );
          const buffer = new Uint8Array( imageData.data.buffer ).buffer; // copy in case the length isn't correct

          this.encode_transform( matrixToAffine( matrix ) );
          this.encode_linewidth( -1 );

          const shape = Shape.rect( 0, 0, canvas.width, canvas.height );

          this.encode_kite_shape( shape, true, true, 100 );

          this.encode_image( new BufferImage( canvas.width, canvas.height, buffer ) );
        }
      }

      node.children.forEach( child => recurse( child ) );

      if ( hasClip ) {
        this.encode_end_clip();
      }

      if ( !node.matrix.isIdentity() ) {
        matrixStack.pop();
      }
    };

    recurse( topNode );
  }
}

scenery.register( 'PhetEncoding', PhetEncoding );
