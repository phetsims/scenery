// Copyright 2023, University of Colorado Boulder

/**
 * Test rasterization
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigIntVector2, BigRational, BigRationalVector2, BoundsIntersectionFilter, IntersectionPoint, PolygonClipping, RenderPathProgram, RenderProgram, scenery } from '../../imports.js';
import { RenderPath } from './RenderProgram.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import Utils from '../../../../dot/js/Utils.js';

class RationalIntersection {
  public constructor( public readonly t: BigRational, public readonly point: BigRationalVector2 ) {}
}

class IntegerEdge {

  public readonly bounds: Bounds2;
  public readonly intersections: RationalIntersection[] = [];

  public constructor(
    public readonly renderPath: RenderPath,
    public readonly x0: number,
    public readonly y0: number,
    public readonly x1: number,
    public readonly y1: number
  ) {
    // TODO: maybe don't compute this here? Can we compute it in the other work?
    this.bounds = new Bounds2(
      Math.min( x0, x1 ),
      Math.min( y0, y1 ),
      Math.max( x0, x1 ),
      Math.max( y0, y1 )
    );
  }
}

class Face {
  public readonly boundary: RationalHalfEdge[] = [];
  public readonly holes: RationalHalfEdge[] = [];
}

class RationalHalfEdge {

  public face: Face | null = null;
  public nextEdge: RationalHalfEdge | null = null;

  public reversed!: RationalHalfEdge; // We will fill this in immediately
  public windingMap = new Map<RenderPath, number>();

  // 0 for straight +x, 1 for +y, 2 for straight -x, 3 for -y
  public discriminator!: number; // filled in immediately

  public slope!: BigRational; // filled in immediately

  public constructor(
    public readonly edgeId: number,
    public readonly p0: BigRationalVector2,
    public readonly p1: BigRationalVector2
  ) {}

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
    this.windingMap.forEach( ( winding, quantity ) => {
      other.windingMap.set( quantity, ( other.windingMap.get( quantity ) || 0 ) + winding );
    } );
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
}

export default class Rasterize {
  public static rasterizeRenderProgram( renderProgram: RenderProgram, bounds: Bounds2 ): void {

    assert && assert( Number.isInteger( bounds.left ) && Number.isInteger( bounds.top ) && Number.isInteger( bounds.right ) && Number.isInteger( bounds.bottom ) );

    // const imageData = new ImageData( bounds.width, bounds.height, { colorSpace: 'srgb' } );

    const scale = Math.pow( 2, 20 - Math.ceil( Math.log2( Math.max( bounds.width, bounds.height ) ) ) );

    const paths: RenderPath[] = [];
    renderProgram.depthFirst( program => {
      if ( program instanceof RenderPathProgram && program.path !== null ) {
        paths.push( program.path );
      }
    } );

    const integerBounds = new Bounds2(
      Utils.roundSymmetric( bounds.minX * scale ),
      Utils.roundSymmetric( bounds.minY * scale ),
      Utils.roundSymmetric( bounds.maxX * scale ),
      Utils.roundSymmetric( bounds.maxY * scale )
    );

    const integerEdges: IntegerEdge[] = [];

    paths.forEach( path => {
      path.subpaths.forEach( subpath => {
        const clippedSubpath = PolygonClipping.boundsClipPolygon( subpath, bounds );

        for ( let i = 0; i < clippedSubpath.length; i++ ) {
          const p0 = clippedSubpath[ i ];
          const p1 = clippedSubpath[ ( i + 1 ) % clippedSubpath.length ];
          const x0 = Utils.roundSymmetric( p0.x * scale );
          const y0 = Utils.roundSymmetric( p0.y * scale );
          const x1 = Utils.roundSymmetric( p1.x * scale );
          const y1 = Utils.roundSymmetric( p1.y * scale );
          integerEdges.push( new IntegerEdge( path, x0, y0, x1, y1 ) );
        }
      } );
    } );

    // Compute intersections
    BoundsIntersectionFilter.quadraticIntersect( integerBounds, integerEdges, ( edgeA, edgeB ) => {
      const intersectionPoints = IntersectionPoint.intersectLineSegments(
        new BigIntVector2( BigInt( edgeA.x0 ), BigInt( edgeA.y0 ) ),
        new BigIntVector2( BigInt( edgeA.x1 ), BigInt( edgeA.y1 ) ),
        new BigIntVector2( BigInt( edgeB.x0 ), BigInt( edgeB.y0 ) ),
        new BigIntVector2( BigInt( edgeB.x1 ), BigInt( edgeB.y1 ) )
      );

      for ( let i = 0; i < intersectionPoints.length; i++ ) {
        const intersectionPoint = intersectionPoints[ i ];

        const t0 = intersectionPoint.t0;
        const t1 = intersectionPoint.t1;
        const point = intersectionPoint.point;

        if ( !t0.equals( BigRational.ZERO ) && !t0.equals( BigRational.ONE ) ) {
          edgeA.intersections.push( new RationalIntersection( t0, point ) );
        }
        if ( !t1.equals( BigRational.ZERO ) && !t1.equals( BigRational.ONE ) ) {
          edgeB.intersections.push( new RationalIntersection( t1, point ) );
        }
      }
    } );

    let edgeIdCounter = 0;
    const rationalHalfEdges: RationalHalfEdge[] = [];
    integerEdges.forEach( integerEdge => {
      const points = [
        new BigRationalVector2( BigRational.whole( integerEdge.x0 ), BigRational.whole( integerEdge.y0 ) )
      ];

      let lastT = BigRational.ZERO;

      integerEdge.intersections.sort( ( a, b ) => {
        // TODO: we'll need to map this over with functions
        return a.t.compareCrossMul( b.t );
      } );

      // Deduplicate
      integerEdge.intersections.forEach( intersection => {
        if ( !lastT.equals( intersection.t ) ) {
          points.push( intersection.point );
        }
        lastT = intersection.t;
      } );

      points.push( ...integerEdge.intersections.map( intersection => intersection.point ) );

      points.push( new BigRationalVector2( BigRational.whole( integerEdge.x1 ), BigRational.whole( integerEdge.y1 ) ) );

      for ( let i = 0; i < points.length; i++ ) {
        const p0 = points[ i ];
        const p1 = points[ ( i + 1 ) % points.length ];

        // We will remove degenerate edges now, so during the deduplication we won't collapse them together
        if ( !p0.equals( p1 ) ) {
          const edgeId = edgeIdCounter++;
          const forwardEdge = new RationalHalfEdge( edgeId, p0, p1 );
          const reverseEdge = new RationalHalfEdge( edgeId, p1, p0 );
          forwardEdge.reversed = reverseEdge;
          reverseEdge.reversed = forwardEdge;
          forwardEdge.windingMap.set( integerEdge.renderPath, 1 );
          reverseEdge.windingMap.set( integerEdge.renderPath, -1 );

          const deltaX = integerEdge.x1 - integerEdge.x0;
          const deltaY = integerEdge.y1 - integerEdge.y0;

          const discriminator = deltaY === 0 ? ( deltaX > 0 ? 0 : 2 ) : ( deltaY > 0 ? 1 : 3 );
          const slope = deltaY === 0 ? BigRational.ZERO : new BigRational( deltaX, deltaY ).reduced();

          forwardEdge.discriminator = discriminator;
          reverseEdge.discriminator = ( discriminator + 2 ) % 4;
          forwardEdge.slope = slope;
          reverseEdge.slope = slope;

          rationalHalfEdges.push( forwardEdge );
          rationalHalfEdges.push( reverseEdge );
        }
      }
    } );

    rationalHalfEdges.sort( ( a, b ) => a.compare( b ) );

    // Do filtering for duplicate half-edges AND connecting edge linked list in the same traversal
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
          lastEdge.nextEdge = edge.reversed;
          lastEdge = edge;
        }
      }
      else {
        lastEdge.nextEdge = firstEdge.reversed;
        filteredRationalHalfEdges.push( edge );
        firstEdge = edge;
        lastEdge = edge;
      }
    }
    lastEdge.reversed.nextEdge = firstEdge; // last connection
  }
}

scenery.register( 'Rasterize', Rasterize );
