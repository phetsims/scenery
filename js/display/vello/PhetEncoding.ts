// Copyright 2023, University of Colorado Boulder

/**
 * A Vello encoding that can draw specific nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import { Affine, Color, Encoding, Extend, GradientStop, LinearGradient, Paint, PaintDef, RadialGradient, scenery, TPaint, VelloColorStop } from '../../imports.js';

const convertColor = ( color: Color ) => {
  return ( ( color.r << 24 ) + ( color.g << 16 ) + ( color.b << 8 ) + ( Math.floor( color.a * 255 ) & 0xff ) ) >>> 0;
};

const convertColorStop = ( color_stop: GradientStop ) => {
  return new VelloColorStop( color_stop.ratio, convertColor( PaintDef.toColor( color_stop.color ) ) );
};

export default class PhetEncoding extends Encoding {

  public encodePaint( paint: TPaint ): void {
    if ( paint instanceof Paint ) {
      if ( paint instanceof LinearGradient ) {
        this.encodeLinearGradient( paint.start.x, paint.start.y, paint.end.x, paint.end.y, paint.stops.map( convertColorStop ), 1, Extend.Pad );
      }
      else if ( paint instanceof RadialGradient ) {
        this.encodeRadialGradient( paint.start.x, paint.start.y, paint.startRadius, paint.end.x, paint.end.y, paint.endRadius, paint.stops.map( convertColorStop ), 1, Extend.Pad );
      }
      else {
        // Pattern, no-op for now
        // TODO: implement pattern, shouldn't be too hard
        // TODO: I said that, didn't I? WE NEED "Extend" to be implemented for images for us to do pattern nicely
        console.log( 'PATTERN UNIMPLEMENTED' );
        this.encodeColor( 0 );
      }
    }
    else {
      const color = PaintDef.toColor( paint );
      this.encodeColor( convertColor( color ) );
    }
  }

  public encodeMatrix( matrix: Matrix3 ): void {
    this.encodeTransform( new Affine( matrix.m00(), matrix.m10(), matrix.m01(), matrix.m11(), matrix.m02(), matrix.m12() ) );
  }
}

scenery.register( 'PhetEncoding', PhetEncoding );
