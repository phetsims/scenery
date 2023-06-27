// Copyright 2023, University of Colorado Boulder

/**
 * A Vello encoding that can draw specific nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { default as wasmInit, load_font_data } from './swash.js';
import Arial from './.Arial.js'; // eslint-disable-line default-import-match-filename
import swash_wasm from './swash_wasm.js';
import Matrix3 from '../../../../dot/js/Matrix3.js';
import { Affine, base64ToU8, Color, Encoding, Extend, GradientStop, LinearGradient, Paint, PaintDef, RadialGradient, scenery, TPaint, VelloColorStop } from '../../imports.js';

let loaded = false;

// TODO: use scenery imports for things to avoid circular reference issues

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

  public encode_matrix( matrix: Matrix3 ): void {
    this.encode_transform( new Affine( matrix.m00(), matrix.m10(), matrix.m01(), matrix.m11(), matrix.m02(), matrix.m12() ) );
  }
}

scenery.register( 'PhetEncoding', PhetEncoding );
