// Copyright 2023, University of Colorado Boulder

/**
 * A ClippableFace from a set of line segment edges. Should still represent multiple closed loops, but it is not
 * explicit.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, ClippableFaceAccumulator, GridClipCallback, LinearEdge, PolygonalFace, PolygonBilinear, PolygonClipping, PolygonCompleteCallback, PolygonMitchellNetravali, scenery, SerializedLinearEdge } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Range from '../../../../../dot/js/Range.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Utils from '../../../../../dot/js/Utils.js';
import { Shape } from '../../../../../kite/js/imports.js';

const scratchVectorA = new Vector2( 0, 0 );
const scratchVectorB = new Vector2( 0, 0 );

export default class EdgedFace implements ClippableFace {
  public constructor( public readonly edges: LinearEdge[] ) {
    // Check on validating edges, since our binary clips won't work well if things aren't matched up (can get extra
    // edges).
    assertSlow && this.validateStartEndMatches();
  }

  /**
   * Converts the face to a polygonal face. Epsilon is used to determine whether start/end points match.
   *
   * NOTE: This is likely a low-performance method, and should only be used for debugging.
   */
  public toPolygonalFace( epsilon = 1e-8 ): PolygonalFace {
    return new PolygonalFace( LinearEdge.toPolygons( this.edges, epsilon ) );
  }

  /**
   * Converts the face to an edged face.
   */
  public toEdgedFace(): EdgedFace {
    return this;
  }

  /**
   * Returns a Shape for the face.
   *
   * NOTE: This is likely a low-performance method, and should only be used for debugging.
   */
  public getShape( epsilon = 1e-8 ): Shape {
    return this.toPolygonalFace( epsilon ).getShape();
  }

  /**
   * Returns the bounds of the face (ignoring any "fake" edges, if the type supports them)
   */
  public getBounds(): Bounds2 {
    const result = Bounds2.NOTHING.copy();
    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      if ( !edge.containsFakeCorner ) {
        result.addPoint( edge.startPoint );
        result.addPoint( edge.endPoint );
      }
    }
    return result;
  }

  /**
   * Returns the range of values for the dot product of the given normal with any point contained within the face
   * (for polygons, this is the same as the range of values for the dot product of the normal with any vertex).
   */
  public getDotRange( normal: Vector2 ): Range {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      // TODO: containsFakeCorner should... propagate with clipping operations, no?
      if ( !edge.containsFakeCorner ) {
        const dotStart = edge.startPoint.dot( normal );
        const dotEnd = edge.endPoint.dot( normal );
        min = Math.min( min, dotStart, dotEnd );
        max = Math.max( max, dotStart, dotEnd );
      }
    }

    return new Range( min, max );
  }

  /**
   * Returns the range of distances from the given point to every point along the edges of the face.
   * For instance, if the face was the unit cube, the range would be 1/2 to sqrt(2), for distances to the middles of
   * the edges and the corners respectively.
   */
  public getDistanceRangeToEdges( point: Vector2 ): Range {
    let min = Number.POSITIVE_INFINITY;
    let max = 0;

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      const p0x = edge.startPoint.x - point.x;
      const p0y = edge.startPoint.y - point.y;
      const p1x = edge.endPoint.x - point.x;
      const p1y = edge.endPoint.y - point.y;

      min = Math.min( min, LinearEdge.evaluateClosestDistanceToOrigin( p0x, p0y, p1x, p1y ) );
      max = Math.max( max, Math.sqrt( p0x * p0x + p0y * p0y ), Math.sqrt( p1x * p1x + p1y * p1y ) );
    }

    return new Range( min, max );
  }

  /**
   * Returns the range of distances from the given point to every point inside the face. The upper bound should be
   * the same as getDistanceRangeToEdges, however the lower bound may be 0 if the point is inside the face.
   */
  public getDistanceRangeToInside( point: Vector2 ): Range {
    const range = this.getDistanceRangeToEdges( point );

    if ( this.containsPoint( point ) ) {
      return new Range( 0, range.max );
    }
    else {
      return range;
    }
  }

  /**
   * Returns the signed area of the face (positive if the vertices are in counter-clockwise order, negative if clockwise)
   */
  public getArea(): number {
    let area = 0;
    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      const p0 = edge.startPoint;
      const p1 = edge.endPoint;

      // Shoelace formula for the area
      area += ( p1.x + p0.x ) * ( p1.y - p0.y );
    }

    return 0.5 * area;
  }

  /**
   * Returns the partial for the centroid computation. These should be summed up, divided by 6, and divided by the area
   * to give the full centroid
   */
  public getCentroidPartial(): Vector2 {
    let x = 0;
    let y = 0;

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      const p0 = edge.startPoint;
      const p1 = edge.endPoint;

      // Partial centroid evaluation. NOTE: using the compound version here, for performance/stability tradeoffs
      const base = ( p0.x * ( 2 * p0.y + p1.y ) + p1.x * ( p0.y + 2 * p1.y ) );
      x += ( p0.x - p1.x ) * base;
      y += ( p1.y - p0.y ) * base;
    }

    return new Vector2( x, y );
  }

  /**
   * Returns the centroid of the face (area is required for the typical integral required to evaluate)
   */
  public getCentroid( area: number ): Vector2 {
    return this.getCentroidPartial().timesScalar( 1 / ( 6 * area ) );
  }

  /**
   * Returns the evaluation of an integral that will be zero if the boundaries of the face are correctly closed.
   * It is designed so that if there is a "gap" and we have open boundaries, the result will likely be non-zero.
   *
   * NOTE: This is only used for debugging, so performance is not a concern.
   */
  public getZero(): number {
    return _.sum( this.edges.map( e => e.getLineIntegralZero() ) );
  }

  /**
   * Returns the average distance from the given point to every point inside the face. The integral evaluation requires
   * the area (similarly to the centroid computation).
   */
  public getAverageDistance( point: Vector2, area: number ): number {
    let sum = 0;

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      const p0 = edge.startPoint;
      const p1 = edge.endPoint;

      sum += LinearEdge.evaluateLineIntegralDistance(
        p0.x - point.x,
        p0.y - point.y,
        p1.x - point.x,
        p1.y - point.y
      );
    }

    return sum / area;
  }

  /**
   * Returns the average distance from the origin to every point inside the face transformed by the given matrix.
   */
  public getAverageDistanceTransformedToOrigin( transform: Matrix3, area: number ): number {
    let sum = 0;

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      const p0 = transform.multiplyVector2( scratchVectorA.set( edge.startPoint ) );
      const p1 = transform.multiplyVector2( scratchVectorB.set( edge.endPoint ) );

      sum += LinearEdge.evaluateLineIntegralDistance( p0.x, p0.y, p1.x, p1.y );
    }

    // We need to account for how much the transform will scale the area
    return sum / ( area * transform.getSignedScale() );
  }

  /**
   * Returns a copy of the face that is clipped to be within the given axis-aligned bounding box.
   */
  public getClipped( minX: number, minY: number, maxX: number, maxY: number ): EdgedFace {
    const edges: LinearEdge[] = [];

    const centerX = ( minX + maxX ) / 2;
    const centerY = ( minY + maxY ) / 2;

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];
      PolygonClipping.boundsClipEdge(
        edge.startPoint, edge.endPoint,
        minX, minY, maxX, maxY, centerX, centerY,
        edges
      );
    }

    return new EdgedFace( edges );
  }

  /**
   * Returns two copies of the face, one that is clipped to be to the left of the given x value, and one that is
   * clipped to be to the right of the given x value.
   *
   * The fakeCornerY is used to determine the "fake" corner that is used for unsorted-edge clipping.
   */
  public getBinaryXClip( x: number, fakeCornerY: number ): { minFace: EdgedFace; maxFace: EdgedFace } {
    const minEdges: LinearEdge[] = [];
    const maxEdges: LinearEdge[] = [];

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      PolygonClipping.binaryXClipEdge( edge.startPoint, edge.endPoint, x, fakeCornerY, minEdges, maxEdges );
    }

    assert && assert( minEdges.every( e => e.startPoint.x <= x && e.endPoint.x <= x ) );
    assert && assert( maxEdges.every( e => e.startPoint.x >= x && e.endPoint.x >= x ) );

    return {
      minFace: new EdgedFace( minEdges ),
      maxFace: new EdgedFace( maxEdges )
    };
  }

  /**
   * Returns two copies of the face, one that is clipped to y values less than the given y value, and one that is
   * clipped to values greater than the given y value.
   *
   * The fakeCornerX is used to determine the "fake" corner that is used for unsorted-edge clipping.
   */
  public getBinaryYClip( y: number, fakeCornerX: number ): { minFace: EdgedFace; maxFace: EdgedFace } {
    const minEdges: LinearEdge[] = [];
    const maxEdges: LinearEdge[] = [];

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      PolygonClipping.binaryYClipEdge( edge.startPoint, edge.endPoint, y, fakeCornerX, minEdges, maxEdges );
    }

    assert && assert( minEdges.every( e => e.startPoint.y <= y && e.endPoint.y <= y ) );
    assert && assert( maxEdges.every( e => e.startPoint.y >= y && e.endPoint.y >= y ) );

    return {
      minFace: new EdgedFace( minEdges ),
      maxFace: new EdgedFace( maxEdges )
    };
  }

  /**
   * Returns two copies of the face, one that is clipped to contain points where dot( normal, point ) < value,
   * and one that is clipped to contain points where dot( normal, point ) > value.
   *
   * The fake corner perpendicular is used to determine the "fake" corner that is used for unsorted-edge clipping
   */
  public getBinaryLineClip( normal: Vector2, value: number, fakeCornerPerpendicular: number ): { minFace: EdgedFace; maxFace: EdgedFace } {
    const minEdges: LinearEdge[] = [];
    const maxEdges: LinearEdge[] = [];

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      PolygonClipping.binaryLineClipEdge( edge.startPoint, edge.endPoint, normal, value, fakeCornerPerpendicular, minEdges, maxEdges );
    }

    assert && assert( minEdges.every( e => normal.dot( e.startPoint ) <= value && normal.dot( e.endPoint ) <= value ) );
    assert && assert( maxEdges.every( e => normal.dot( e.startPoint ) >= value && normal.dot( e.endPoint ) >= value ) );

    return {
      minFace: new EdgedFace( minEdges ),
      maxFace: new EdgedFace( maxEdges )
    };
  }

  /**
   * Returns an array of faces, clipped similarly to getBinaryLineClip, but with more than one (parallel) split line at
   * a time. The first face will be the one with dot( normal, point ) < values[0], the second one with
   * values[ 0 ] < dot( normal, point ) < values[1], etc.
   */
  public getStripeLineClip( normal: Vector2, values: number[], fakeCornerPerpendicular: number ): EdgedFace[] {
    if ( values.length === 0 ) {
      return [ this ];
    }

    const edgesCollection: LinearEdge[][] = _.range( values.length + 1 ).map( () => [] );

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      PolygonClipping.binaryStripeClipEdge( edge.startPoint, edge.endPoint, normal, values, fakeCornerPerpendicular, edgesCollection );
    }

    if ( assert ) {
      for ( let i = 0; i < edgesCollection.length; i++ ) {
        const edges = edgesCollection[ i ];

        const minValue = i > 0 ? values[ i - 1 ] : Number.NEGATIVE_INFINITY;
        const maxValue = i < values.length ? values[ i ] : Number.POSITIVE_INFINITY;

        assert( edges.every( e => {
          return normal.dot( e.startPoint ) + 1e-8 >= minValue && normal.dot( e.startPoint ) - 1e-8 <= maxValue &&
                 normal.dot( e.endPoint ) + 1e-8 >= minValue && normal.dot( e.endPoint ) - 1e-8 <= maxValue;
        } ) );
      }
    }

    return edgesCollection.map( edges => new EdgedFace( edges ) );
  }

  /**
   * Returns two copies of the face, one that is clipped to contain points inside the circle defined by the given
   * center and radius, and one that is clipped to contain points outside the circle.
   *
   * NOTE: maxAngleSplit is used to determine the polygonal approximation of the circle. The returned result will not
   * have a chord with an angle greater than maxAngleSplit.
   */
  public getBinaryCircularClip( center: Vector2, radius: number, maxAngleSplit: number ): { insideFace: EdgedFace; outsideFace: EdgedFace } {
    const insideEdges: LinearEdge[] = [];
    const outsideEdges: LinearEdge[] = [];

    PolygonClipping.binaryCircularClipEdges( this.edges, center, radius, maxAngleSplit, insideEdges, outsideEdges );

    return {
      insideFace: new EdgedFace( insideEdges ),
      outsideFace: new EdgedFace( outsideEdges )
    };
  }

  /**
   * Given an integral bounding box and step sizes (which define the grid), this will clip the face to each cell in the
   * grid, calling the callback for each cell's contributing edges (in order, if we are a PolygonalFace).
   * polygonCompleteCallback will be called whenever a polygon is completed (if we are a polygonal type of face).
   */
  public gridClipIterate(
    minX: number, minY: number, maxX: number, maxY: number,
    stepX: number, stepY: number, stepWidth: number, stepHeight: number,
    callback: GridClipCallback,
    polygonCompleteCallback: PolygonCompleteCallback
  ): void {
    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      PolygonClipping.gridClipIterate(
        edge.startPoint, edge.endPoint,
        minX, minY, maxX, maxY,
        stepX, stepY, stepWidth, stepHeight,
        callback
      );
    }

    if ( this.edges.length ) {
      polygonCompleteCallback();
    }
  }

  /**
   * Returns the evaluation of the bilinear (tent) filter integrals for the given point, ASSUMING that the face
   * is clipped to the transformed unit square of x: [minX,minX+1], y: [minY,minY+1].
   */
  public getBilinearFiltered( pointX: number, pointY: number, minX: number, minY: number ): number {
    return PolygonBilinear.evaluateLinearEdges( this.edges, pointX, pointY, minX, minY );
  }

  /**
   * Returns the evaluation of the Mitchell-Netravali (1/3,1/3) filter integrals for the given point, ASSUMING that the
   * face is clipped to the transformed unit square of x: [minX,minX+1], y: [minY,minY+1].
   */
  public getMitchellNetravaliFiltered( pointX: number, pointY: number, minX: number, minY: number ): number {
    return PolygonMitchellNetravali.evaluateLinearEdges( this.edges, pointX, pointY, minX, minY );
  }

  /**
   * Returns whether the face contains the given point.
   */
  public containsPoint( point: Vector2 ): boolean {
    return LinearEdge.getWindingNumberEdges( this.edges, point ) !== 0;
  }

  /**
   * Returns an affine-transformed version of the face.
   */
  public getTransformed( transform: Matrix3 ): EdgedFace {
    if ( transform.isIdentity() ) {
      return this;
    }
    else {
      const transformedEdges: LinearEdge[] = [];

      for ( let i = 0; i < this.edges.length; i++ ) {
        const edge = this.edges[ i ];

        const start = transform.timesVector2( edge.startPoint );
        const end = transform.timesVector2( edge.endPoint );

        if ( !start.equals( end ) ) {
          transformedEdges.push( new LinearEdge( start, end ) );
        }
      }

      return new EdgedFace( transformedEdges );
    }
  }

  /**
   * Returns a rounded version of the face, where [-epsilon/2, epsilon/2] rounds to 0, etc.
   */
  public getRounded( epsilon: number ): EdgedFace {
    const edges = [];

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      const startPoint = new Vector2(
        Utils.roundSymmetric( edge.startPoint.x / epsilon ) * epsilon,
        Utils.roundSymmetric( edge.startPoint.y / epsilon ) * epsilon
      );

      const endPoint = new Vector2(
        Utils.roundSymmetric( edge.endPoint.x / epsilon ) * epsilon,
        Utils.roundSymmetric( edge.endPoint.y / epsilon ) * epsilon
      );

      if ( !startPoint.equals( endPoint ) ) {
        edges.push( new LinearEdge( startPoint, endPoint, edge.containsFakeCorner ) );
      }
    }

    return new EdgedFace( edges );
  }

  /**
   * Calls the callback with points for each edge in the face.
   */
  public forEachEdge( callback: ( startPoint: Vector2, endPoint: Vector2 ) => void ): void {
    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];
      callback( edge.startPoint, edge.endPoint );
    }
  }

  /**
   * Returns a singleton accumulator for this type of face.
   */
  public getScratchAccumulator(): ClippableFaceAccumulator {
    return scratchAccumulator;
  }

  /**
   * Returns a new accumulator for this type of face.
   */
  public getAccumulator(): ClippableFaceAccumulator {
    return new EdgedFaceAccumulator();
  }

  /**
   * Returns a debugging string.
   */
  public toString(): string {
    return this.edges.map( e => `${e.startPoint.x},${e.startPoint.y} => ${e.endPoint.x},${e.endPoint.y}` ).join( '\n' );
  }

  /**
   * Returns a serialized version of the face, that should be able to be deserialized into the same type of face.
   * See {FaceType}.deserialize.
   *
   * NOTE: If you don't know what type of face this is, use serializeClippableFace instead.
   */
  public serialize(): SerializedEdgedFace {
    return {
      edges: this.edges.map( edge => edge.serialize() )
    };
  }

  public validateStartEndMatches(): void {
    if ( assertSlow ) {
      assertSlow( Math.abs( this.getZero() ) < 1e-5, 'Ensure we are effectively closed' );

      // Ensure that each point's 'starts' and 'ends' matches precisely
      type Entry = { point: Vector2; startCount: number; endCount: number };
      const entries: Entry[] = [];
      const getEntry = ( point: Vector2 ): Entry => {
        for ( let i = 0; i < entries.length; i++ ) {
          if ( entries[ i ].point.equals( point ) ) {
            return entries[ i ];
          }
        }
        const entry = { point: point, startCount: 0, endCount: 0 };
        entries.push( entry );
        return entry;
      };
      for ( let i = 0; i < this.edges.length; i++ ) {
        const edge = this.edges[ i ];
        getEntry( edge.startPoint ).startCount++;
        getEntry( edge.endPoint ).endCount++;
      }
      for ( let i = 0; i < entries.length; i++ ) {
        const entry = entries[ i ];
        assertSlow( entry.startCount === entry.endCount, 'Ensure each point has matching start/end counts' );
      }
    }
  }

  public static deserialize( serialized: SerializedEdgedFace ): EdgedFace {
    return new EdgedFace( serialized.edges.map( edge => LinearEdge.deserialize( edge ) ) );
  }

  public static fromBounds( bounds: Bounds2 ): EdgedFace {
    return EdgedFace.fromBoundsValues( bounds.minX, bounds.minY, bounds.maxX, bounds.maxY );
  }

  public static fromBoundsValues( minX: number, minY: number, maxX: number, maxY: number ): EdgedFace {
    const p0 = new Vector2( minX, minY );
    const p1 = new Vector2( maxX, minY );
    const p2 = new Vector2( maxX, maxY );
    const p3 = new Vector2( minX, maxY );

    return new EdgedFace( [
      new LinearEdge( p0, p1 ),
      new LinearEdge( p1, p2 ),
      new LinearEdge( p2, p3 ),
      new LinearEdge( p3, p0 )
    ] );
  }
}

scenery.register( 'EdgedFace', EdgedFace );

export class EdgedFaceAccumulator implements ClippableFaceAccumulator {

  private edges: LinearEdge[] = [];

  public addEdge( startX: number, startY: number, endX: number, endY: number, startPoint: Vector2 | null, endPoint: Vector2 | null ): void {
    this.edges.push( new LinearEdge(
      startPoint || new Vector2( startX, startY ),
      endPoint || new Vector2( endX, endY )
    ) );
  }

  public markNewPolygon(): void {
    // no-op, since we're storing unsorted edges!
  }

  // Will reset it to the initial state also
  public finalizeFace(): EdgedFace | null {
    if ( this.edges.length === 0 ) {
      return null;
    }

    const edges = this.edges;
    this.edges = [];
    return new EdgedFace( edges );
  }

  // Will reset without creating a face
  public reset(): void {
    this.edges.length = 0;
  }
}

const scratchAccumulator = new EdgedFaceAccumulator();

export type SerializedEdgedFace = {
  edges: SerializedLinearEdge[];
};
