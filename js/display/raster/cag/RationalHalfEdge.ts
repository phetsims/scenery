// Copyright 2023, University of Colorado Boulder

/**
 * Represents a half-edge (directed line segment) with rational coordinates.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigRational, BigRationalVector2, RationalBoundary, RationalFace, scenery, WindingMap } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

// Instead of storing vertices, we can get away with storing half-edges, with a linked list of next/previous and the
// opposite half edge. This is like a half-edge winged data structure.
export default class RationalHalfEdge {

  public face: RationalFace | null = null;
  public nextEdge: RationalHalfEdge | null = null;
  public previousEdge: RationalHalfEdge | null = null; // exists so we can enumerate edges at a vertex
  public boundary: RationalBoundary | null = null;

  public reversed!: RationalHalfEdge; // We will fill this in immediately
  public windingMap = new WindingMap();

  // 0 for straight +x, 1 for +y, 2 for straight -x, 3 for -y
  public discriminator!: number; // filled in immediately

  public slope!: BigRational; // filled in immediately

  public p0float: Vector2;
  public p1float: Vector2;

  public processed = false; // used in some algorithms!

  public constructor(
    public readonly edgeId: number,
    public readonly p0: BigRationalVector2,
    public readonly p1: BigRationalVector2
  ) {
    this.p0float = new Vector2( p0.x.toFloat(), p0.y.toFloat() );
    this.p1float = new Vector2( p1.x.toFloat(), p1.y.toFloat() );
  }

  // See LinearEdge.leftComparison
  private leftComparison( x: BigRational, y: BigRational ): BigRational {
    // TODO: estimate how many bits we will need!
    // ( p1x - p0x ) * ( y - p0y ) - ( x - p0x ) * ( p1y - p0y );
    return this.p1.x.minus( this.p0.x ).times( y.minus( this.p0.y ) ).minus( x.minus( this.p0.x ).times( this.p1.y.minus( this.p0.y ) ) );
  }

  // See LinearEdge.windingContribution
  public windingContribution( x: BigRational, y: BigRational ): number {
    const cmp0 = this.p0.y.compareCrossMul( y );
    const cmp1 = this.p1.y.compareCrossMul( y );

    if ( cmp0 <= 0 ) {
      // If it's an upward crossing and P is to the left of the edge
      if ( cmp1 > 0 && this.leftComparison( x, y ).isPositive() ) {
        return 1; // have a valid "up" intersection
      }
    }
    else { // p0y > y (no test needed)
      // If it's a downward crossing and P is to the right of the edge
      if ( cmp1 <= 0 && this.leftComparison( x, y ).isNegative() ) {
        return -1; // have a valid "down" intersection
      }
    }

    return 0;
  }

  public static compareBigInt( a: bigint, b: bigint ): number {
    return a < b ? -1 : ( a > b ? 1 : 0 );
  }

  // Provides a stable comparison, but this is NOT numerical!!!
  public static quickCompareBigRational( a: BigRational, b: BigRational ): number {
    const numeratorCompare = RationalHalfEdge.compareBigInt( a.numerator, b.numerator );
    if ( numeratorCompare !== 0 ) {
      return numeratorCompare;
    }
    return RationalHalfEdge.compareBigInt( a.denominator, b.denominator );
  }

  public static quickCompareBigRationalVector2( a: BigRationalVector2, b: BigRationalVector2 ): number {
    const xCompare = RationalHalfEdge.quickCompareBigRational( a.x, b.x );
    if ( xCompare !== 0 ) {
      return xCompare;
    }
    return RationalHalfEdge.quickCompareBigRational( a.y, b.y );
  }

  public addWindingFrom( other: RationalHalfEdge ): void {
    this.windingMap.addWindingMap( other.windingMap );
  }

  public compare( other: RationalHalfEdge ): number {
    // can have an arbitrary sort for the first point
    const p0Compare = RationalHalfEdge.quickCompareBigRationalVector2( this.p0, other.p0 );
    if ( p0Compare !== 0 ) {
      return p0Compare;
    }

    // now an angle-based sort for the second point
    if ( this.discriminator < other.discriminator ) {
      return -1;
    }
    else if ( this.discriminator > other.discriminator ) {
      return 1;
    }
    // NOTE: using x/y "slope", so it's a bit inverted
    const slopeCompare = this.slope.compareCrossMul( other.slope );
    if ( slopeCompare !== 0 ) {
      return -slopeCompare;
    }

    // Now, we're sorting "identically overlapping" half-edges
    return this.edgeId < other.edgeId ? -1 : ( this.edgeId > other.edgeId ? 1 : 0 );
  }

  public static filterAndConnectHalfEdges( rationalHalfEdges: RationalHalfEdge[] ): RationalHalfEdge[] {
    // Do filtering for duplicate half-edges AND connecting edge linked list in the same traversal
    // NOTE: We don't NEED to filter "low-order" vertices (edge whose opposite is its next edge), but we could at
    // some point in the future. Note that removing a low-order edge then might create ANOTHER low-order edge, so
    // it would need to chase these.
    // NOTE: We could also remove "composite" edges that have no winding contribution (degenerate "touching" in the
    // source path), however it's probably not too common so it's not done here.
    let firstEdge = rationalHalfEdges[ 0 ];
    let lastEdge = rationalHalfEdges[ 0 ];
    const filteredRationalHalfEdges = [ lastEdge ];
    for ( let i = 1; i < rationalHalfEdges.length; i++ ) {
      const edge = rationalHalfEdges[ i ];

      if ( edge.p0.equals( lastEdge.p0 ) ) {
        if ( edge.p1.equals( lastEdge.p1 ) ) {
          lastEdge.addWindingFrom( edge );
        }
        else {
          filteredRationalHalfEdges.push( edge );
          edge.reversed.nextEdge = lastEdge;
          lastEdge.previousEdge = edge.reversed;
          lastEdge = edge;
        }
      }
      else {
        firstEdge.reversed.nextEdge = lastEdge;
        lastEdge.previousEdge = firstEdge.reversed;
        filteredRationalHalfEdges.push( edge );
        firstEdge = edge;
        lastEdge = edge;
      }
    }
    // last connection
    firstEdge.reversed.nextEdge = lastEdge;
    lastEdge.previousEdge = firstEdge.reversed;
    return filteredRationalHalfEdges;
  }
}

scenery.register( 'RationalHalfEdge', RationalHalfEdge );
