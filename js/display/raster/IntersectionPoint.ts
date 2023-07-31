// Copyright 2023, University of Colorado Boulder

/**
 * Intersection of two line segments, with t0 and t1 representing the interpolation values for the two segments.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigIntVector2, BigRational, BigRationalVector2, scenery } from '../../imports.js';

export default class IntersectionPoint {
  public constructor( public t0: BigRational, public t1: BigRational, public point: BigRationalVector2 ) {}

  public toString(): string {
    return `t0=${this.t0}, t1=${this.t1}, point=${this.point}`;
  }

  public verify( p0: BigIntVector2, p1: BigIntVector2, p2: BigIntVector2, p3: BigIntVector2 ): void {
    const px0 = BigRational.whole( p0.x ).plus( this.t0.times( BigRational.whole( p1.x - p0.x ) ) );
    const py0 = BigRational.whole( p0.y ).plus( this.t0.times( BigRational.whole( p1.y - p0.y ) ) );
    const px1 = BigRational.whole( p2.x ).plus( this.t1.times( BigRational.whole( p3.x - p2.x ) ) );
    const py1 = BigRational.whole( p2.y ).plus( this.t1.times( BigRational.whole( p3.y - p2.y ) ) );
    if ( !px0.equals( px1 ) || !py0.equals( py1 ) ) {
      throw new Error( 'Intersection point does not match' );
    }
  }

  public static intersectLineSegments( p0: BigIntVector2, p1: BigIntVector2, p2: BigIntVector2, p3: BigIntVector2 ): IntersectionPoint[] {
    const p0x = p0.x;
    const p0y = p0.y;
    const p1x = p1.x;
    const p1y = p1.y;
    const p2x = p2.x;
    const p2y = p2.y;
    const p3x = p3.x;
    const p3y = p3.y;

    const d0x = p1x - p0x;
    const d0y = p1y - p0y;
    const d1x = p3x - p2x;
    const d1y = p3y - p2y;

    const cdx = p2x - p0x;
    const cdy = p2y - p0y;

    const denominator = d0x * d1y - d0y * d1x;

    if ( denominator === 0n ) {
      // such that p0 + t * ( p1 - p0 ) = p2 + ( a * t + b ) * ( p3 - p2 )
      // an equivalency between lines
      let a;
      let b;

      const d1x_zero = d1x === 0n;
      const d1y_zero = d1y === 0n;

      // if ( d0s === 0 || d1s === 0 ) {
      //   return NO_OVERLAP;
      // }
      //
      // a = d0s / d1s;
      // b = ( p0s - p2s ) / d1s;

      // TODO: can we reduce the branching here?
      // Find a dimension where our line is not degenerate (e.g. covers multiple values in that dimension)
      // Compute line equivalency there
      if ( d1x_zero && d1y_zero ) {
        // DEGENERATE case for second line, it's a point, bail out
        return [];
      }
      else if ( d1x_zero ) {
        // if d1x is zero AND our denominator is zero, that means d0x or d1y must be zero. We checked d1y above, so d0x must be zero
        if ( p0.x !== p2.x ) {
          // vertical lines, BUT not same x, so no intersection
          return [];
        }
        a = new BigRational( d0y, d1y );
        b = new BigRational( -cdy, d1y );
      }
      else if ( d1y_zero ) {
        // if d1y is zero AND our denominator is zero, that means d0y or d1x must be zero. We checked d1x above, so d0y must be zero
        if ( p0.y !== p2.y ) {
          // horizontal lines, BUT not same y, so no intersection
          return [];
        }
        a = new BigRational( d0x, d1x );
        b = new BigRational( -cdx, d1x );
      }
      else {
        // we have non-axis-aligned second line, use that to compute a,b for each dimension, and we're the same "line"
        // iff those are consistent
        if ( d0x === 0n && d0y === 0n ) {
          // DEGENERATE first line, it's a point, bail out
          return [];
        }
        const ax = new BigRational( d0x, d1x );
        const ay = new BigRational( d0y, d1y );
        if ( !ax.equalsCrossMul( ay ) ) {
          return [];
        }
        const bx = new BigRational( -cdx, d1x );
        const by = new BigRational( -cdy, d1y );
        if ( !bx.equalsCrossMul( by ) ) {
          return [];
        }

        // Pick the one with a non-zero a, so it is invertible
        if ( ax.isZero() ) {
          a = ay;
          b = by;
        }
        else {
          a = ax;
          b = bx;
        }
      }

      const points = [];

      // p0 + t * ( p1 - p0 ) = p2 + ( a * t + b ) * ( p3 - p2 )
      // i.e. line0( t ) = line1( a * t + b )
      // replacements for endpoints:
      // t=0       =>  t0=0,        t1=b
      // t=1       =>  t0=1,        t1=a+b
      // t=-b/a    =>  t0=-b/a,     t1=0
      // t=(1-b)/a =>  t0=(1-b)/a,  t1=1

      // NOTE: cases become identical if b=0, b=1, b=-a, b=1-a, HOWEVER these would not be internal, so they would be
      // excluded, and we can ignore them

      // t0=0, t1=b, p0
      const case1t1 = b;
      if ( case1t1.ratioTest() === 2 ) {
        const p = new IntersectionPoint( BigRational.ZERO, case1t1.reduced(), new BigRationalVector2( BigRational.whole( p0x ), BigRational.whole( p0y ) ) );
        p.verify( p0, p1, p2, p3 );
        points.push( p );
      }

      // t0=1, t1=a+b, p1
      const case2t1 = new BigRational( a.numerator + b.numerator, a.denominator ); // abuse a,b having same denominator
      if ( case2t1.ratioTest() === 2 ) {
        const p = new IntersectionPoint( BigRational.ONE, case2t1.reduced(), new BigRationalVector2( BigRational.whole( p1x ), BigRational.whole( p1y ) ) );
        p.verify( p0, p1, p2, p3 );
        points.push( p );
      }

      // t0=-b/a, t1=0, p2
      const case3t0 = new BigRational( -b.numerator, a.numerator ); // abuse a,b having same denominator
      if ( case3t0.ratioTest() === 2 ) {
        const p = new IntersectionPoint( case3t0.reduced(), BigRational.ZERO, new BigRationalVector2( BigRational.whole( p2x ), BigRational.whole( p2y ) ) );
        p.verify( p0, p1, p2, p3 );
        points.push( p );
      }

      // t0=(1-b)/a, t1=1, p3
      // ( 1 - b ) / a = ( denom - b_numer ) / denom / ( a_numer / denom ) = ( denom - b_numer ) / a_numer
      const case4t0 = new BigRational( a.denominator - b.numerator, a.numerator );
      if ( case4t0.ratioTest() === 2 ) {
        const p = new IntersectionPoint( case4t0.reduced(), BigRational.ONE, new BigRationalVector2( BigRational.whole( p3x ), BigRational.whole( p3y ) ) );
        p.verify( p0, p1, p2, p3 );
        points.push( p );
      }

      return points;
    }
    else {
      const t_numerator = cdx * d1y - cdy * d1x;
      const u_numerator = cdx * d0y - cdy * d0x;

      // This will move the sign to the numerator, BUT won't do the reduction (let us first see if there is an intersection)
      const t_raw = new BigRational( t_numerator, denominator );
      const u_raw = new BigRational( u_numerator, denominator );

      // 2i means totally internal, 1i means on an endpoint, 0i means totally external
      const t_cmp = t_raw.ratioTest();
      const u_cmp = u_raw.ratioTest();

      if ( t_cmp <= 0n || u_cmp <= 0n ) {
        return []; // outside one or both segments
      }
      else if ( t_cmp === 1 && u_cmp === 1 ) {
        return []; // on endpoints of both segments (we ignore that, we only want something internal to one)
      }
      else {
        // use parametric segment definition to get the intersection point
        // x0 + t * (x1 - x0)
        // p0x + t_numerator / denominator * d0x
        // ( denominator * p0x + t_numerator * d0x ) / denominator
        const x_numerator = denominator * p0x + t_numerator * d0x;
        const y_numerator = denominator * p0y + t_numerator * d0y;

        const x_raw = new BigRational( x_numerator, denominator );
        const y_raw = new BigRational( y_numerator, denominator );

        const x = x_raw.reduced();
        const y = y_raw.reduced();

        const t = t_raw.reduced();
        const u = u_raw.reduced();

        // NOTE: will t/u be exactly 0,1 for endpoints if they are endpoints, no?
        const point = new IntersectionPoint( t, u, new BigRationalVector2( x, y ) );
        point.verify( p0, p1, p2, p3 );
        return [ point ];
      }
    }
  }
}

scenery.register( 'IntersectionPoint', IntersectionPoint );
