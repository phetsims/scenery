// Copyright 2023, University of Colorado Boulder

/**
 * Contains an import-style snippet of shader code, with dependencies on other snippets.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';
import Vector2 from '../../../../dot/js/Vector2.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';

type Code = number;

const X_MAX_CODE = 0x1;
const Y_MAX_CODE = 0x2;
const X_MIN_CODE = 0x4;
const Y_MIN_CODE = 0x8;
const TWO_BITS_CODE = 0x10;

class CodedVector2 extends Vector2 {
  public constructor( x: number, y: number, public code: Code ) {
    super( x, y );
  }

  public override copy(): CodedVector2 {
    return new CodedVector2( this.x, this.y, this.code );
  }

  public toVector2(): Vector2 {
    return new Vector2( this.x, this.y );
  }

  public updateCode( bounds: Bounds2 ): void {
    this.code = CohenSutherlandClipping.getCode( this.x, this.y, bounds );
  }

  public static create( p: Vector2, bounds: Bounds2 ): CodedVector2 {
    return new CodedVector2( p.x, p.y, CohenSutherlandClipping.getCode( p.x, p.y, bounds ) );
  }
}

export default class CohenSutherlandClipping {

  // Mutates the vectors, returns whether the line segment is at least partially within the bounds
  private static clip( p0: CodedVector2, p1: CodedVector2, bounds: Bounds2 ): boolean {
    while ( ( p0.code | p1.code ) !== 0 ) {
      if ( ( p0.code & p1.code & ~TWO_BITS_CODE ) !== 0 ) {
        return false;
      }

      // Choose the first point not in the window
      const c = p0.code === 0 ? p1.code : p0.code;

      let x: number;
      let y: number;

      // Now clip against line corresponding to first nonzero bit
      if ( ( X_MIN_CODE & c ) !== 0 ) {
        x = bounds.left;
        y = p0.y + ( p1.y - p0.y ) * ( bounds.left - p0.x ) / ( p1.x - p0.x );
      }
      else if ( ( X_MAX_CODE & c ) !== 0 ) {
        x = bounds.right;
        y = p0.y + ( p1.y - p0.y ) * ( bounds.right - p0.x ) / ( p1.x - p0.x );
      }
      else if ( ( Y_MIN_CODE & c ) !== 0 ) {
        x = p0.x + ( p1.x - p0.x ) * ( bounds.top - p0.y ) / ( p1.y - p0.y );
        y = bounds.top;
      }
      else if ( ( Y_MAX_CODE & c ) !== 0 ) {
        x = p0.x + ( p1.x - p0.x ) * ( bounds.bottom - p0.y ) / ( p1.y - p0.y );
        y = bounds.bottom;
      }
      else {
        throw new Error( 'cohenSutherlandClip: Unknown case' );
      }
      if ( c === p0.code ) {
        p0.x = x;
        p0.y = y;
        p0.updateCode( bounds );
      }
      else {
        p1.x = x;
        p1.y = y;
        p1.updateCode( bounds );
      }
    }

    return true;
  }

  // The Maillot extension of the Cohen-Sutherland encoding of points
  public static getCode( x: number, y: number, bounds: Bounds2 ): Code {
    if ( x < bounds.minX ) {
      if ( y > bounds.maxY ) {
        return 0x16;
      }
      else if ( y < bounds.minY ) {
        return 0x1c;
      }
      else {
        return 0x4;
      }
    }
    else if ( x > bounds.maxX ) {
      if ( y > bounds.maxY ) {
        return 0x13;
      }
      else if ( y < bounds.minY ) {
        return 0x19;
      }
      else {
        return 0x1;
      }
    }
    else if ( y > bounds.maxY ) {
      return 0x2;
    }
    else if ( y < bounds.minY ) {
      return 0x8;
    }
    else {
      return 0;
    }
  }
}

scenery.register( 'CohenSutherlandClipping', CohenSutherlandClipping );
