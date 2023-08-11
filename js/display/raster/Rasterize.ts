// Copyright 2023, University of Colorado Boulder

/**
 * Test rasterization
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigIntVector2, BigRational, BigRationalVector2, BoundsIntersectionFilter, IntersectionPoint, PolygonClipping, RenderColor, RenderPathProgram, RenderProgram, scenery } from '../../imports.js';
import { RenderPath } from './RenderProgram.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import Utils from '../../../../dot/js/Utils.js';
import Vector2 from '../../../../dot/js/Vector2.js';
import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import Vector4 from '../../../../dot/js/Vector4.js';

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

class WindingMap {
  public constructor( public readonly map: Map<RenderPath, number> = new Map() ) {}

  public getWindingNumber( renderPath: RenderPath ): number {
    return this.map.get( renderPath ) || 0;
  }

  public addWindingNumber( renderPath: RenderPath, amount: number ): void {
    const current = this.getWindingNumber( renderPath );
    this.map.set( renderPath, current + amount );
  }

  public addWindingMap( windingMap: WindingMap ): void {
    for ( const [ renderPath, winding ] of windingMap.map ) {
      this.addWindingNumber( renderPath, winding );
    }
  }

  public equals( windingMap: WindingMap ): boolean {
    if ( this.map.size !== windingMap.map.size ) {
      return false;
    }
    for ( const [ renderPath, winding ] of this.map ) {
      if ( winding !== windingMap.getWindingNumber( renderPath ) ) {
        return false;
      }
    }
    return true;
  }
}

// Instead of storing vertices, we can get away with storing half-edges, with a linked list of next/previous and the
// opposite half edge. This is like a half-edge winged data structure.
class RationalHalfEdge {

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

  public constructor(
    public readonly edgeId: number,
    public readonly p0: BigRationalVector2,
    public readonly p1: BigRationalVector2
  ) {
    this.p0float = new Vector2( p0.x.toFloat(), p0.y.toFloat() );
    this.p1float = new Vector2( p1.x.toFloat(), p1.y.toFloat() );
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
}

class RationalBoundary {
  public readonly edges: RationalHalfEdge[] = [];
  public signedArea!: number;
  public bounds!: Bounds2;
  public minimalXRationalPoint!: BigRationalVector2;

  public computeProperties(): void {
    let signedArea = 0;
    const bounds = Bounds2.NOTHING.copy();
    let minimalXP0Edge = this.edges[ 0 ];

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i % this.edges.length ];

      if ( edge.p0.x.compareCrossMul( minimalXP0Edge.p0.x ) < 0 ) {
        minimalXP0Edge = edge;
      }

      const p0float = edge.p0float;
      const p1float = edge.p1float;

      bounds.addPoint( p0float );

      // PolygonIntegrals.evaluateShoelaceArea( p0.x, p0.y, p1.x, p1.y );
      signedArea += 0.5 * ( p1float.x + p0float.x ) * ( p1float.y - p0float.y );
    }

    this.minimalXRationalPoint = minimalXP0Edge.p0;
    this.bounds = bounds;
    this.signedArea = signedArea;
  }

  public toTransformedPolygon( scaleFactor = 1, translation = Vector2.ZERO ): Vector2[] {
    const result: Vector2[] = [];
    for ( let i = 0; i < this.edges.length; i++ ) {
      result.push( this.edges[ i ].p0float.timesScalar( scaleFactor ).plus( translation ) );
    }
    return result;
  }
}

// Relies on the main boundary being positive-oriented, and the holes being negative-oriented and non-overlapping
class PolygonalFace {
  public constructor( public readonly polygons: Vector2[][] ) {}

  public getArea(): number {
    let area = 0;
    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];

      // TODO: optimize more?
      for ( let j = 0; j < polygon.length; j++ ) {
        const p0 = polygon[ j ];
        const p1 = polygon[ ( j + 1 ) % polygon.length ];
        // PolygonIntegrals.evaluateShoelaceArea( p0.x, p0.y, p1.x, p1.y );
        area += ( p1.x + p0.x ) * ( p1.y - p0.y );
      }
    }

    return 0.5 * area;
  }

  public getCentroid( area: number ): Vector2 {
    let x = 0;
    let y = 0;

    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];

      // TODO: optimize more?
      for ( let j = 0; j < polygon.length; j++ ) {
        const p0 = polygon[ j ];
        const p1 = polygon[ ( j + 1 ) % polygon.length ];

        // evaluateCentroidPartial
        const base = ( 1 / 6 ) * ( p0.x * p1.y - p1.x * p0.y );
        x += ( p0.x + p1.x ) * base;
        y += ( p0.y + p1.y ) * base;
      }
    }

    return new Vector2(
      x / area,
      y / area
    );
  }

  public getClipped( bounds: Bounds2 ): PolygonalFace {
    return new PolygonalFace( this.polygons.map( polygon => PolygonClipping.boundsClipPolygon( polygon, bounds ) ) );
  }
}

class RationalFace {
  public readonly holes: RationalBoundary[] = [];
  public windingMapMap = new Map<RationalFace, WindingMap>();
  public windingMap: WindingMap | null = null;
  public inclusionSet: Set<RenderPath> = new Set<RenderPath>();
  public renderProgram: RenderProgram | null = null;

  public constructor( public readonly boundary: RationalBoundary ) {}
}

export default class Rasterize {
  public static rasterizeRenderProgram( renderProgram: RenderProgram, bounds: Bounds2 ): Record<string, IntentionalAny> | null {

    let debugData: Record<string, IntentionalAny> | null;
    if ( assert ) {
      debugData = {
        areas: []
      };
    }

    assert && assert( Number.isInteger( bounds.left ) && Number.isInteger( bounds.top ) && Number.isInteger( bounds.right ) && Number.isInteger( bounds.bottom ) );

    const scale = Math.pow( 2, 20 - Math.ceil( Math.log2( Math.max( bounds.width, bounds.height ) ) ) );
    if ( assert ) {
      debugData!.scale = scale;
    }

    const paths: RenderPath[] = [];
    renderProgram.depthFirst( program => {
      if ( program instanceof RenderPathProgram && program.path !== null ) {
        paths.push( program.path );
      }
    } );
    const backgroundPath = new RenderPath( 'nonzero', [
      [
        bounds.leftTop,
        bounds.rightTop,
        bounds.rightBottom,
        bounds.leftBottom
      ]
    ] );
    paths.push( backgroundPath );

    const integerBounds = new Bounds2(
      Utils.roundSymmetric( bounds.minX * scale ),
      Utils.roundSymmetric( bounds.minY * scale ),
      Utils.roundSymmetric( bounds.maxX * scale ),
      Utils.roundSymmetric( bounds.maxY * scale )
    );
    if ( assert ) {
      debugData!.integerBounds = integerBounds;
    }

    const integerEdges: IntegerEdge[] = [];
    if ( assert ) {
      debugData!.integerEdges = integerEdges;
    }

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

        // TODO: in WGSL, use atomicExchange to write a linked list of these into each edge
        // NOTE: We filter out endpoints of lines, since they wouldn't trigger a split in the segment anyway
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

      points.push( new BigRationalVector2( BigRational.whole( integerEdge.x1 ), BigRational.whole( integerEdge.y1 ) ) );

      for ( let i = 0; i < points.length - 1; i++ ) {
        const p0 = points[ i ];
        const p1 = points[ i + 1 ];

        // We will remove degenerate edges now, so during the deduplication we won't collapse them together
        if ( !p0.equals( p1 ) ) {
          const edgeId = edgeIdCounter++;
          const forwardEdge = new RationalHalfEdge( edgeId, p0, p1 );
          const reverseEdge = new RationalHalfEdge( edgeId, p1, p0 );
          forwardEdge.reversed = reverseEdge;
          reverseEdge.reversed = forwardEdge;
          forwardEdge.windingMap.addWindingNumber( integerEdge.renderPath, 1 );
          reverseEdge.windingMap.addWindingNumber( integerEdge.renderPath, -1 );

          const deltaX = integerEdge.x1 - integerEdge.x0;
          const deltaY = integerEdge.y1 - integerEdge.y0;

          // We compute slope here due to rational precision (while it would be possible to create a larger rational
          // number later and reduce it, here we're starting with integers, so we don't have to do as much).
          const discriminator = deltaY === 0 ? ( deltaX > 0 ? 0 : 2 ) : ( deltaY > 0 ? 1 : 3 );
          const slope = deltaY === 0 ? BigRational.ZERO : new BigRational( deltaX, deltaY ).reduced();

          // We store the slope and discriminator here, as that allows us to tell the order-difference between two
          // edges that have one point the same. This works here, because we have already broken lines up at the
          // endpoints in the case of overlap, so that if it has the same start point, discriminator and slope, then it
          // WILL have the same end point, and thus will be the same effective edge.
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
    // NOTE: We don't NEED to filter "low-order" vertices (edge whose opposite is its next edge), but we could at
    // some point in the future. Note that removing a low-order edge then might create ANOTHER low-order edge, so
    // it would need to chase these.
    // NOTE: We could also remove "composite" edges that have no winding contribution (degenerate "touching" in the
    // source path), however it's probably not too common so it's not done here.
    let firstEdge = rationalHalfEdges[ 0 ];
    let lastEdge = rationalHalfEdges[ 0 ];
    const filteredRationalHalfEdges = [ lastEdge ];
    if ( assert ) {
      debugData!.filteredRationalHalfEdges = filteredRationalHalfEdges;
    }
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

    const innerBoundaries: RationalBoundary[] = [];
    const outerBoundaries: RationalBoundary[] = [];
    const faces: RationalFace[] = [];
    if ( assert ) {
      debugData!.innerBoundaries = innerBoundaries;
      debugData!.outerBoundaries = outerBoundaries;
      debugData!.faces = faces;
    }
    for ( let i = 0; i < filteredRationalHalfEdges.length; i++ ) {
      const firstEdge = filteredRationalHalfEdges[ i ];
      if ( !firstEdge.boundary ) {
        const boundary = new RationalBoundary();
        boundary.edges.push( firstEdge );
        firstEdge.boundary = boundary;

        let edge = firstEdge.nextEdge!;
        while ( edge !== firstEdge ) {
          edge.boundary = boundary;
          boundary.edges.push( edge );
          edge = edge.nextEdge!;
        }

        boundary.computeProperties();

        const signedArea = boundary.signedArea;
        if ( Math.abs( signedArea ) > 1e-8 ) {
          if ( signedArea > 0 ) {
            innerBoundaries.push( boundary );
            const face = new RationalFace( boundary );
            faces.push( face );
            for ( let j = 0; j < boundary.edges.length; j++ ) {
              const edge = boundary.edges[ j ];
              edge.face = face;
            }
          }
          else {
            outerBoundaries.push( boundary );
          }
        }
      }
    }

    // Compute which boundaries are holes in which faces
    let exteriorBoundary: RationalBoundary | null = null;
    if ( assert ) {
      debugData!.exteriorBoundary = exteriorBoundary;
    }
    for ( let i = 0; i < outerBoundaries.length; i++ ) {
      const outerBoundary = outerBoundaries[ i ];
      const outerBounds = outerBoundary.bounds;

      const boundaryDebugData: IntentionalAny = assert ? {
        outerBoundary: outerBoundary
      } : null;
      if ( assert ) {
        debugData!.boundaryDebugData = debugData!.boundaryDebugData || [];
        debugData!.boundaryDebugData.push( boundaryDebugData );
      }

      const minimalRationalPoint = outerBoundary.minimalXRationalPoint;

      let maxIntersectionX = new BigRational( integerBounds.left - 1, 1 );
      let maxIntersectionEdge: RationalHalfEdge | null = null;
      let maxIntersectionIsVertex = false;

      for ( let j = 0; j < faces.length; j++ ) {
        const face = faces[ j ];
        const innerBoundary = face.boundary;
        const innerBounds = innerBoundary.bounds;

        // Check if the "inner" bounds actually fully contains (strictly) our "outer" bounds.
        // This is a constraint that has to be satisfied for the outer boundary to be a hole.
        if (
          outerBounds.minX > innerBounds.minX &&
          outerBounds.minY > innerBounds.minY &&
          outerBounds.maxX < innerBounds.maxX &&
          outerBounds.maxY < innerBounds.maxY
        ) {
          for ( let k = 0; k < innerBoundary.edges.length; k++ ) {
            const edge = innerBoundary.edges[ k ];

            // TODO: This will require a lot of precision, how do we handle this?
            // TODO: we'll need to handle these anyway!
            const dx0 = edge.p0.x.minus( minimalRationalPoint.x );
            const dx1 = edge.p1.x.minus( minimalRationalPoint.x );

            // If both x values of the segment are at or to the right, there will be no intersection
            if ( dx0.isNegative() || dx1.isNegative() ) {

              const dy0 = edge.p0.y.minus( minimalRationalPoint.y );
              const dy1 = edge.p1.y.minus( minimalRationalPoint.y );

              const bothPositive = dy0.isPositive() && dy1.isPositive();
              const bothNegative = dy0.isNegative() && dy1.isNegative();

              if ( !bothPositive && !bothNegative ) {
                const isZero0 = dy0.isZero();
                const isZero1 = dy1.isZero();

                let candidateMaxIntersectionX: BigRational;
                let isVertex: boolean;
                if ( isZero0 && isZero1 ) {
                  // NOTE: on a vertex
                  const is0Less = edge.p0.x.compareCrossMul( edge.p1.x ) < 0;
                  candidateMaxIntersectionX = is0Less ? edge.p1.x : edge.p0.x;
                  isVertex = true;
                }
                else if ( isZero0 ) {
                  // NOTE: on a vertex
                  candidateMaxIntersectionX = edge.p0.x;
                  isVertex = true;
                }
                else if ( isZero1 ) {
                  // NOTE: on a vertex
                  candidateMaxIntersectionX = edge.p1.x;
                  isVertex = true;
                }
                else {
                  // p0.x + ( p1.x - p0.x ) * ( minimalRationalPoint.y - p0.y ) / ( p1.y - p0.y );
                  // TODO: could simplify by reversing sign and using dy1
                  candidateMaxIntersectionX = edge.p0.x.plus( edge.p1.x.minus( edge.p0.x ).times( minimalRationalPoint.y.minus( edge.p0.y ) ).dividedBy( edge.p1.y.minus( edge.p0.y ) ) );
                  isVertex = false;
                }

                // TODO: add less-than, etc.
                if ( maxIntersectionX.compareCrossMul( candidateMaxIntersectionX ) < 0 ) {
                  maxIntersectionX = candidateMaxIntersectionX;
                  maxIntersectionEdge = edge;
                  maxIntersectionIsVertex = isVertex;
                }
              }
            }
          }
        }
      }

      if ( assert ) {
        boundaryDebugData.maxIntersectionX = maxIntersectionX;
        boundaryDebugData.maxIntersectionEdge = maxIntersectionEdge;
        boundaryDebugData.maxIntersectionIsVertex = maxIntersectionIsVertex;
      }

      let connectedFace: RationalFace | null = null;
      if ( maxIntersectionEdge ) {
        const edge0 = maxIntersectionEdge;
        const edge1 = maxIntersectionEdge.reversed;
        if ( !edge0.face ) {
          connectedFace = edge1.face!;
        }
        else if ( !edge1.face ) {
          connectedFace = edge0.face!;
        }
        else if ( maxIntersectionIsVertex ) {
          // We'll need to traverse around the vertex to find the face we need.

          // Get a starting edge with p0 = intersection
          const startEdge = ( edge0.p0.x.equalsCrossMul( maxIntersectionX ) && edge0.p0.y.equalsCrossMul( minimalRationalPoint.y ) ) ? edge0 : edge1;

          assert && assert( startEdge.p0.x.equalsCrossMul( maxIntersectionX ) );
          assert && assert( startEdge.p0.y.equalsCrossMul( minimalRationalPoint.y ) );

          // TODO: for testing this, remember we'll need multiple "fully surrounding" boundaries?
          // TODO: wait, no we won't
          let bestEdge = startEdge;
          let edge = startEdge.previousEdge!.reversed;
          while ( edge !== startEdge ) {
            if ( edge.compare( bestEdge ) < 0 ) {
              bestEdge = edge;
            }
            edge = edge.previousEdge!.reversed;
          }
          connectedFace = edge.face!; // TODO: why do we NOT reverse it here?!? reversed issues?
        }
        else {
          // non-vertex, a bit easier
          // TODO: could grab this value stored from earlier
          const isP0YLess = edge0.p0.y.compareCrossMul( edge0.p1.y ) < 0;
          // Because it should have a "positive" orientation, we want the "negative-y-facing edge"
          connectedFace = isP0YLess ? edge1.face : edge0.face;
        }

        assert && assert( connectedFace );
        connectedFace.holes.push( outerBoundary );
      }
      else {
        exteriorBoundary = outerBoundary;
      }

      if ( assert ) {
        boundaryDebugData.connectedFace = connectedFace;
      }
    }

    // Fill in face data for holes, so we can traverse nicely
    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];

      for ( let j = 0; j < face.holes.length; j++ ) {
        const hole = face.holes[ j ];

        for ( let k = 0; k < hole.edges.length; k++ ) {
          const edge = hole.edges[ k ];

          edge.face = face;
        }
      }
    }

    // For ease of use, an unbounded face (it is essentially fake)
    const unboundedFace = new RationalFace( exteriorBoundary! );
    if ( assert ) {
      debugData!.unboundedFace = unboundedFace;
    }
    for ( let i = 0; i < exteriorBoundary!.edges.length; i++ ) {
      const edge = exteriorBoundary!.edges[ i ];
      edge.face = unboundedFace;
    }

    for ( let i = 0; i < filteredRationalHalfEdges.length; i++ ) {
      const edge = filteredRationalHalfEdges[ i ];

      const face = edge.face!;
      const otherFace = edge.reversed.face!;

      assert && assert( face );
      assert && assert( otherFace );

      // TODO: possibly reverse this, check to see which winding map is correct
      if ( !face.windingMapMap.has( otherFace ) ) {
        face.windingMapMap.set( otherFace, edge.windingMap );
      }
    }

    unboundedFace.windingMap = new WindingMap(); // no windings, empty!
    const recursiveWindingMap = ( solvedFace: RationalFace ) => {
      // TODO: no recursion, could blow recursion limits
      for ( const [ otherFace, windingMap ] of solvedFace.windingMapMap ) {
        const needsNewWindingMap = !otherFace.windingMap;

        if ( needsNewWindingMap || assert ) {
          const newWindingMap = new WindingMap();
          const existingMap = solvedFace.windingMap!;
          const deltaMap = windingMap;

          newWindingMap.addWindingMap( existingMap );
          newWindingMap.addWindingMap( deltaMap );

          if ( assert ) {
            // TODO: object for the winding map?
          }
          otherFace.windingMap = newWindingMap;

          if ( needsNewWindingMap ) {
            recursiveWindingMap( otherFace );
          }
        }
      }
    };
    recursiveWindingMap( unboundedFace );

    const rasterWidth = bounds.width;
    const rasterHeight = bounds.height;

    const accumulationBuffer: Vector4[] = [];
    if ( assert ) {
      debugData!.accumulationBuffer = accumulationBuffer;
    }
    for ( let i = 0; i < rasterWidth * rasterHeight; i++ ) {
      accumulationBuffer.push( Vector4.ZERO.copy() );
    }

    const inverseScale = 1 / scale;
    const translation = new Vector2( -bounds.minX, -bounds.minY );

    for ( let i = 0; i < faces.length; i++ ) {
      const face = faces[ i ];

      const faceDebugData: IntentionalAny = assert ? {
        face: face,
        pixels: [],
        areas: []
      } : null;
      if ( assert ) {
        debugData!.faceDebugData = debugData!.faceDebugData || [];
        debugData!.faceDebugData.push( faceDebugData );
      }

      face.inclusionSet = new Set<RenderPath>();
      for ( const renderPath of face.windingMap!.map.keys() ) {
        const windingNumber = face.windingMap!.getWindingNumber( renderPath );
        const included = renderPath.fillRule === 'nonzero' ? windingNumber !== 0 : windingNumber % 2 !== 0;
        if ( included ) {
          face.inclusionSet.add( renderPath );
        }
      }
      const faceRenderProgram = renderProgram.simplify( renderPath => face.inclusionSet.has( renderPath ) );
      face.renderProgram = faceRenderProgram;

      // Drop faces that will be fully transparent
      if ( faceRenderProgram instanceof RenderColor && faceRenderProgram.color.w <= 1e-8 ) {
        continue;
      }

      // Now back in our normal coordinate frame!
      const polygonalFace = new PolygonalFace( [
        face.boundary.toTransformedPolygon( inverseScale, translation ),
        ...face.holes.map( hole => hole.toTransformedPolygon( inverseScale, translation ) )
      ] );
      if ( assert ) {
        faceDebugData.polygonalFace = polygonalFace;
      }

      // We'll avoid calculating this from the fresh polygon data, so we can avoid the performance cost
      // (for when this is WGSL'ed).
      const polygonalBounds = Bounds2.NOTHING.copy();
      polygonalBounds.includeBounds( face.boundary.bounds );
      for ( let j = 0; j < face.holes.length; j++ ) {
        polygonalBounds.includeBounds( face.holes[ j ].bounds );
      }
      polygonalBounds.minX = polygonalBounds.minX * inverseScale + translation.x;
      polygonalBounds.minY = polygonalBounds.minY * inverseScale + translation.y;
      polygonalBounds.maxX = polygonalBounds.maxX * inverseScale + translation.x;
      polygonalBounds.maxY = polygonalBounds.maxY * inverseScale + translation.y;
      if ( assert ) {
        faceDebugData.polygonalBounds = polygonalBounds;
      }

      const minX = Math.max( Math.floor( polygonalBounds.minX ), 0 );
      const minY = Math.max( Math.floor( polygonalBounds.minY ), 0 );
      const maxX = Math.min( Math.ceil( polygonalBounds.maxX ), rasterWidth );
      const maxY = Math.min( Math.ceil( polygonalBounds.maxY ), rasterHeight );

      const constColor = faceRenderProgram instanceof RenderColor ? faceRenderProgram.color : null;

      const addPixel = ( pixelFace: PolygonalFace, area: number, x: number, y: number ) => {
        if ( area > 1e-8 ) {
          if ( assert ) {
            debugData!.areas.push( new Bounds2( x, y, x + 1, y + 1 ) );
          }
          const index = y * rasterWidth + x;

          let color;
          if ( constColor ) {
            color = constColor;
          }
          else {
            const centroid = pixelFace.getCentroid( area ).minus( translation );
            color = faceRenderProgram.evaluate( centroid );
          }
          accumulationBuffer[ index ].add( color.timesScalar( area ) );
        }
      };

      const scratchVector = new Vector2( 0, 0 );
      const addFullArea = ( face: PolygonalFace, minX: number, minY: number, maxX: number, maxY: number ) => {
        if ( assert ) {
          debugData!.areas.push( new Bounds2( minX, minY, maxX, maxY ) );
        }
        if ( constColor ) {
          for ( let y = minY; y < maxY; y++ ) {
            for ( let x = minX; x < maxX; x++ ) {
              accumulationBuffer[ y * rasterWidth + x ].add( constColor );
            }
          }
        }
        else {
          for ( let y = minY; y < maxY; y++ ) {
            for ( let x = minX; x < maxX; x++ ) {
              const centroid = scratchVector.setXY( x + 0.5, y + 0.5 );
              accumulationBuffer[ y * rasterWidth + x ].add( faceRenderProgram.evaluate( centroid ) );
            }
          }
        }
      };

      // TODO: don't shadow
      const binaryRender = ( face: PolygonalFace, area: number, minX: number, minY: number, maxX: number, maxY: number ) => {
        const xDiff = maxX - minX;
        const yDiff = maxY - minY;
        if ( xDiff === 1 && yDiff === 1 ) {
          addPixel( face, area, minX, minY );
        }
        else if ( area >= ( maxX - minX ) * ( maxY - minY ) - 1e-8 ) {
          addFullArea( face, minX, minY, maxX, maxY );
        }
        else {
          if ( xDiff > yDiff ) {
            const xSplit = Math.floor( ( minX + maxX ) / 2 );
            // TODO: allocation?
            const leftFace = face.getClipped( new Bounds2( minX, minY, xSplit, maxY ) );
            const rightFace = face.getClipped( new Bounds2( xSplit, minY, maxX, maxY ) );

            const leftArea = leftFace.getArea();
            const rightArea = rightFace.getArea();

            if ( leftArea > 1e-8 ) {
              binaryRender( leftFace, leftArea, minX, minY, xSplit, maxY );
            }
            if ( rightArea > 1e-8 ) {
              binaryRender( rightFace, rightArea, xSplit, minY, maxX, maxY );
            }
          }
          else {
            const ySplit = Math.floor( ( minY + maxY ) / 2 );
            // TODO: allocation?
            const topFace = face.getClipped( new Bounds2( minX, minY, maxX, ySplit ) );
            const bottomFace = face.getClipped( new Bounds2( minX, ySplit, maxX, maxY ) );

            const topArea = topFace.getArea();
            const bottomArea = bottomFace.getArea();

            if ( topArea > 1e-8 ) {
              binaryRender( topFace, topArea, minX, minY, maxX, ySplit );
            }
            if ( bottomArea > 1e-8 ) {
              binaryRender( bottomFace, bottomArea, minX, ySplit, maxX, maxY );
            }
          }
        }
      };

      const fullRender = () => {
        const pixelBounds = Bounds2.NOTHING.copy();
        for ( let y = minY; y < maxY; y++ ) {
          pixelBounds.minY = y;
          pixelBounds.maxY = y + 1;
          for ( let x = minX; x < maxX; x++ ) {
            pixelBounds.minX = x;
            pixelBounds.maxX = x + 1;

            const pixelFace = polygonalFace.getClipped( pixelBounds );
            // if ( assert ) {
            //   const area = pixelFace.getArea();
            //   if ( Math.abs( area ) > 1e-8 ) {
            //     faceDebugData.pixels.push( {
            //       x: x,
            //       y: y,
            //       pixelBounds: pixelBounds.copy(),
            //       pixelFace: pixelFace,
            //       area: area,
            //       centroid: pixelFace.getCentroid( area )
            //     } );
            //   }
            // }
            const area = pixelFace.getArea();
            addPixel( pixelFace, area, x, y );
          }
        }
      };

      binaryRender( polygonalFace, polygonalFace.getArea(), minX, minY, maxX, maxY );


      // TODO: more advanced handling

      // TODO: potential filtering!!!

      // TODO TODO TODO TODO TODO: non-zero-centered bounds! Verify everything
    }

    const imageData = new ImageData( rasterWidth, rasterHeight, { colorSpace: 'srgb' } );
    if ( assert ) {
      debugData!.imageData = imageData;
    }

    for ( let i = 0; i < accumulationBuffer.length; i++ ) {
      const accumulation = accumulationBuffer[ i ];
      let x = accumulation.x;
      let y = accumulation.y;
      let z = accumulation.z;
      const a = accumulation.w;

      // unpremultiply
      if ( a > 0 ) {
        x /= a;
        y /= a;
        z /= a;
      }

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

    if ( assert ) {
      const canvas = document.createElement( 'canvas' );
      canvas.width = rasterWidth;
      canvas.height = rasterHeight;
      const context = canvas.getContext( '2d' )!;
      context.putImageData( imageData, 0, 0 );
      debugData!.canvas = canvas;
    }

    return ( debugData! ) || null;
  }
}

scenery.register( 'Rasterize', Rasterize );
