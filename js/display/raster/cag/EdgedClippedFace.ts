// Copyright 2023, University of Colorado Boulder

/**
 * A ClippableFace based on a set of line segment edges, where (a) it is contained within a bounding box, and (b)
 * line segments going along the full border of the bounding box (from one corner to another) will be counted
 * separately. This helps with performance, since EdgedFace on its own would build up large counts of these edges
 * that "undo" each other during recursive clipping operations.
 *
 * Should still represent multiple closed loops, but it is not explicit.
 *
 * "implicit" edges/vertices are those defined by the clipped counts (minXCount, etc.)
 * "explicit" edges/vertices are those in the edges list
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, ClippableFaceAccumulator, EdgedFace, GridClipCallback, LinearEdge, PolygonalFace, PolygonBilinear, PolygonClipping, PolygonCompleteCallback, PolygonMitchellNetravali, scenery, SerializedLinearEdge } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Range from '../../../../../dot/js/Range.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Utils from '../../../../../dot/js/Utils.js';
import { Shape } from '../../../../../kite/js/imports.js';

const scratchVectorA = new Vector2( 0, 0 );
const scratchVectorB = new Vector2( 0, 0 );
const scratchVectorC = new Vector2( 0, 0 );
const scratchVectorD = new Vector2( 0, 0 );

const emptyArray: LinearEdge[] = [];

export default class EdgedClippedFace implements ClippableFace {

  public readonly clippedEdgedFace: EdgedFace;

  public constructor(
    // Should contain only "internal" edges, not those clipped edges that are corner-to-corner along the edge of the
    // bounding box.
    public readonly edges: LinearEdge[],

    // Bounding box
    public readonly minX: number,
    public readonly minY: number,
    public readonly maxX: number,
    public readonly maxY: number,

    // Count of edges from (minX,minY) to (minX,maxY) minus count of edges from (minX,maxY) to (minX,minY)
    // (minX, minY=>maxY positive)
    public readonly minXCount: number,

    // Count of edges from (minX,minY) to (maxX,minY) minus count of edges from (maxX,minY) to (minX,minY)
    // (minX=>maxX positive, minY)
    public readonly minYCount: number,

    // Count of edges from (maxX,minY) to (maxX,maxY) minus count of edges from (maxX,maxY) to (maxX,minY)
    // (maxX, minY=>maxY positive)
    public readonly maxXCount: number,

    // Count of edges from (minX,maxY) to (maxX,maxY) minus count of edges from (maxX,maxY) to (minX,maxY)
    // (minX=>maxX positive, maxY)
    public readonly maxYCount: number
  ) {
    assert && assert( isFinite( minX ) && isFinite( maxX ) && minX <= maxX );
    assert && assert( isFinite( minY ) && isFinite( maxY ) && minY <= maxY );
    assert && assert( isFinite( minXCount ) && Number.isInteger( minXCount ) );
    assert && assert( isFinite( minYCount ) && Number.isInteger( minYCount ) );
    assert && assert( isFinite( maxXCount ) && Number.isInteger( maxXCount ) );
    assert && assert( isFinite( maxYCount ) && Number.isInteger( maxYCount ) );

    assertSlow && assertSlow( edges.every( edge => {
      return edge.startPoint.x >= minX && edge.startPoint.x <= maxX &&
             edge.startPoint.y >= minY && edge.startPoint.y <= maxY &&
             edge.endPoint.x >= minX && edge.endPoint.x <= maxX &&
             edge.endPoint.y >= minY && edge.endPoint.y <= maxY;
    } ) );

    this.clippedEdgedFace = new EdgedFace( edges, true );

    // Check on validating edges, since our binary clips won't work well if things aren't matched up (can get extra
    // edges).
    assertSlow && LinearEdge.validateStartEndMatches( this.getAllEdges() );
  }

  // TODO: also have FAST conversions to here, where we DO NOT scan those edges?

  /**
   * Converts the face to a polygonal face. Epsilon is used to determine whether start/end points match.
   *
   * NOTE: This is likely a low-performance method, and should only be used for debugging.
   */
  public toPolygonalFace( epsilon = 1e-8 ): PolygonalFace {
    // We'll need to add in our counted edges
    return this.toEdgedFace().toPolygonalFace( epsilon );
  }

  /**
   * Converts the face to an edged face.
   */
  public toEdgedFace(): EdgedFace {
    return new EdgedFace( this.getAllEdges() );
  }

  public static fromEdges( edges: LinearEdge[], minX: number, minY: number, maxX: number, maxY: number ): EdgedClippedFace {
    scratchAccumulator.reset();
    for ( let i = 0; i < edges.length; i++ ) {
      const edge = edges[ i ];
      scratchAccumulator.addEdge( edge.startPoint.x, edge.startPoint.y, edge.endPoint.x, edge.endPoint.y, edge.startPoint, edge.endPoint );
    }

    return scratchAccumulator.finalizeEnsureFace( minX, minY, maxX, maxY );
  }

  public static fromEdgesWithNoCheck( edges: LinearEdge[], minX: number, minY: number, maxX: number, maxY: number ): EdgedClippedFace {
    return new EdgedClippedFace( edges, minX, minY, maxX, maxY, 0, 0, 0, 0 );
  }

  private static implicitEdge( startPoint: Vector2, endPoint: Vector2, count: number ): LinearEdge {
    assert && assert( count !== 0 );
    return new LinearEdge(
      count > 0 ? startPoint : endPoint,
      count > 0 ? endPoint : startPoint
    );
  }

  public forEachImplicitEdge( callback: ( startPoint: Vector2, endPoint: Vector2 ) => void ): void {
    const minXMinY = ( this.minXCount || this.minYCount ) ? this.getMinXMinY() : null;
    const minXMaxY = ( this.minXCount || this.maxYCount ) ? this.getMinXMaxY() : null;
    const maxXMinY = ( this.maxXCount || this.minYCount ) ? this.getMaxXMinY() : null;
    const maxXMaxY = ( this.maxXCount || this.maxYCount ) ? this.getMaxXMaxY() : null;

    for ( let i = 0; i !== this.minXCount; i += Math.sign( this.minXCount ) ) {
      assert && assert( minXMinY && minXMaxY );
      this.minXCount > 0 ? callback( minXMinY!, minXMaxY! ) : callback( minXMaxY!, minXMinY! );
    }
    for ( let i = 0; i !== this.minYCount; i += Math.sign( this.minYCount ) ) {
      assert && assert( minXMinY && maxXMinY );
      this.minYCount > 0 ? callback( minXMinY!, maxXMinY! ) : callback( maxXMinY!, minXMinY! );
    }
    for ( let i = 0; i !== this.maxXCount; i += Math.sign( this.maxXCount ) ) {
      assert && assert( maxXMinY && maxXMaxY );
      this.maxXCount > 0 ? callback( maxXMinY!, maxXMaxY! ) : callback( maxXMaxY!, maxXMinY! );
    }
    for ( let i = 0; i !== this.maxYCount; i += Math.sign( this.maxYCount ) ) {
      assert && assert( minXMaxY && maxXMaxY );
      this.maxYCount > 0 ? callback( minXMaxY!, maxXMaxY! ) : callback( maxXMaxY!, minXMaxY! );
    }
  }

  public getImplicitEdges(): LinearEdge[] {
    const edges: LinearEdge[] = [];

    this.forEachImplicitEdge( ( startPoint, endPoint ) => edges.push( new LinearEdge( startPoint, endPoint ) ) );

    return edges;
  }

  public getAllEdges(): LinearEdge[] {
    return [
      ...this.edges,
      ...this.getImplicitEdges()
    ];
  }

  /**
   * Returns a Shape for the face.
   *
   * NOTE: This is likely a low-performance method, and should only be used for debugging.
   */
  public getShape( epsilon = 1e-8 ): Shape {
    return this.toPolygonalFace( epsilon ).getShape();
  }

  public getMinXMinY(): Vector2 {
    return new Vector2( this.minX, this.minY );
  }

  public getMinXMaxY(): Vector2 {
    return new Vector2( this.minX, this.maxY );
  }

  public getMaxXMinY(): Vector2 {
    return new Vector2( this.maxX, this.minY );
  }

  public getMaxXMaxY(): Vector2 {
    return new Vector2( this.maxX, this.maxY );
  }

  /**
   * Returns whether this face has an implicit vertex at the minX-minY corner.
   */
  public hasMinXMinY(): boolean {
    return this.minXCount !== 0 || this.minYCount !== 0;
  }

  /**
   * Returns whether this face has an implicit vertex at the minX-maxY corner.
   */
  public hasMinXMaxY(): boolean {
    return this.minXCount !== 0 || this.maxYCount !== 0;
  }

  /**
   * Returns whether this face has an implicit vertex at the maxX-minY corner.
   */
  public hasMaxXMinY(): boolean {
    return this.maxXCount !== 0 || this.minYCount !== 0;
  }

  /**
   * Returns whether this face has an implicit vertex at the maxX-maxY corner.
   */
  public hasMaxXMaxY(): boolean {
    return this.maxXCount !== 0 || this.maxYCount !== 0;
  }

  /**
   * Returns whether this face has an implicit vertex with minX.
   */
  public hasMinX(): boolean {
    return this.minXCount !== 0 || this.minYCount !== 0 || this.maxYCount !== 0;
  }

  /**
   * Returns whether this face has an implicit vertex with minY.
   */
  public hasMinY(): boolean {
    return this.minYCount !== 0 || this.minXCount !== 0 || this.maxXCount !== 0;
  }

  /**
   * Returns whether this face has an implicit vertex with maxX.
   */
  public hasMaxX(): boolean {
    return this.maxXCount !== 0 || this.minYCount !== 0 || this.maxYCount !== 0;
  }

  /**
   * Returns whether this face has an implicit vertex with maxY.
   */
  public hasMaxY(): boolean {
    return this.maxYCount !== 0 || this.minXCount !== 0 || this.maxXCount !== 0;
  }

  /**
   * Returns the bounds of the face (ignoring any "fake" edges, if the type supports them)
   */
  public getBounds(): Bounds2 {
    const result = this.clippedEdgedFace.getBounds();

    this.hasMinX() && result.addX( this.minX );
    this.hasMinY() && result.addY( this.minY );
    this.hasMaxX() && result.addX( this.maxX );
    this.hasMaxY() && result.addY( this.maxY );

    return result;
  }

  /**
   * Returns the range of values for the dot product of the given normal with any point contained within the face
   * (for polygons, this is the same as the range of values for the dot product of the normal with any vertex).
   */
  public getDotRange( normal: Vector2 ): Range {
    const range = this.clippedEdgedFace.getDotRange( normal );

    this.hasMinXMinY() && range.addValue( normal.x * this.minX + normal.y * this.minY );
    this.hasMinXMaxY() && range.addValue( normal.x * this.minX + normal.y * this.maxY );
    this.hasMaxXMinY() && range.addValue( normal.x * this.maxX + normal.y * this.minY );
    this.hasMaxXMaxY() && range.addValue( normal.x * this.maxX + normal.y * this.maxY );

    return range;
  }

  /**
   * Returns the range of distances from the given point to every point along the edges of the face.
   * For instance, if the face was the unit cube, the range would be 1/2 to sqrt(2), for distances to the middles of
   * the edges and the corners respectively.
   */
  public getDistanceRangeToEdges( point: Vector2 ): Range {
    const range = this.clippedEdgedFace.getDistanceRangeToEdges( point );

    this.minXCount && LinearEdge.addDistanceRange( this.getMinXMinY(), this.getMinXMaxY(), point, range );
    this.minYCount && LinearEdge.addDistanceRange( this.getMinXMinY(), this.getMaxXMinY(), point, range );
    this.maxXCount && LinearEdge.addDistanceRange( this.getMaxXMinY(), this.getMaxXMaxY(), point, range );
    this.maxYCount && LinearEdge.addDistanceRange( this.getMinXMaxY(), this.getMaxXMaxY(), point, range );

    return range;
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
    let area = this.clippedEdgedFace.getArea();

    // NOTE: This ASSUMES that we're using the specific shoelace formulation of ( p1.x + p0.x ) * ( p1.y - p0.y ) in
    // the super call.
    // Our minYCount/maxYCount won't contribute (since they have the same Y values, their shoelace contribution will be
    // zero.
    // ALSO: there is a doubling and non-doubling that cancel out here (1/2 from shoelace, 2* due to x+x).
    area += ( this.maxY - this.minY ) * ( this.minXCount * this.minX + this.maxXCount * this.maxX );

    return area;
  }

  /**
   * Returns the partial for the centroid computation. These should be summed up, divided by 6, and divided by the area
   * to give the full centroid
   */
  public getCentroidPartial(): Vector2 {
    const centroidPartial = this.clippedEdgedFace.getCentroidPartial();

    // NOTE: This ASSUMES we're using the compound formulation, based on
    // xc = ( p0.x - p1.x ) * ( p0.x * ( 2 * p0.y + p1.y ) + p1.x * ( p0.y + 2 * p1.y ) )
    // yc = ( p1.y - p0.y ) * ( p0.x * ( 2 * p0.y + p1.y ) + p1.x * ( p0.y + 2 * p1.y ) )
    if ( this.minYCount || this.maxYCount ) {
      centroidPartial.x += 3 * ( this.minX - this.maxX ) * ( this.minX + this.maxX ) * ( this.minYCount * this.minY + this.maxYCount * this.maxY );
    }
    if ( this.minXCount || this.maxXCount ) {
      centroidPartial.y += 3 * ( this.maxY - this.minY ) * ( this.minY + this.maxY ) * ( this.minXCount * this.minX + this.maxXCount * this.maxX );
    }

    return centroidPartial;
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
    return _.sum( this.getAllEdges().map( e => e.getLineIntegralZero() ) );
  }

  /**
   * Returns the average distance from the given point to every point inside the face. The integral evaluation requires
   * the area (similarly to the centroid computation).
   */
  public getAverageDistance( point: Vector2, area: number ): number {
    let average = this.clippedEdgedFace.getAverageDistance( point, area );

    const minX = this.minX - point.x;
    const minY = this.minY - point.y;
    const maxX = this.maxX - point.x;
    const maxY = this.maxY - point.y;

    if ( this.minXCount ) {
      average += this.minXCount * LinearEdge.evaluateLineIntegralDistance( minX, minY, minX, maxY ) / area;
    }
    if ( this.minYCount ) {
      average += this.minYCount * LinearEdge.evaluateLineIntegralDistance( minX, minY, maxX, minY ) / area;
    }
    if ( this.maxXCount ) {
      average += this.maxXCount * LinearEdge.evaluateLineIntegralDistance( maxX, minY, maxX, maxY ) / area;
    }
    if ( this.maxYCount ) {
      average += this.maxYCount * LinearEdge.evaluateLineIntegralDistance( minX, maxY, maxX, maxY ) / area;
    }

    return average;
  }

  /**
   * Returns the average distance from the origin to every point inside the face transformed by the given matrix.
   */
  public getAverageDistanceTransformedToOrigin( transform: Matrix3, area: number ): number {
    let average = this.clippedEdgedFace.getAverageDistanceTransformedToOrigin( transform, area );

    if ( this.minXCount || this.minYCount || this.maxXCount || this.maxYCount ) {
      const divisor = area * transform.getSignedScale();

      const minXMinY = transform.multiplyVector2( scratchVectorA.setXY( this.minX, this.minY ) );
      const minXMaxY = transform.multiplyVector2( scratchVectorB.setXY( this.minX, this.maxY ) );
      const maxXMinY = transform.multiplyVector2( scratchVectorC.setXY( this.maxX, this.minY ) );
      const maxXMaxY = transform.multiplyVector2( scratchVectorD.setXY( this.maxX, this.maxY ) );

      if ( this.minXCount ) {
        average += this.minXCount * LinearEdge.evaluateLineIntegralDistance( minXMinY.x, minXMinY.y, minXMaxY.x, minXMaxY.y ) / divisor;
      }
      if ( this.minYCount ) {
        average += this.minYCount * LinearEdge.evaluateLineIntegralDistance( minXMinY.x, minXMinY.y, maxXMinY.x, maxXMinY.y ) / divisor;
      }
      if ( this.maxXCount ) {
        average += this.maxXCount * LinearEdge.evaluateLineIntegralDistance( maxXMinY.x, maxXMinY.y, maxXMaxY.x, maxXMaxY.y ) / divisor;
      }
      if ( this.maxYCount ) {
        average += this.maxYCount * LinearEdge.evaluateLineIntegralDistance( minXMaxY.x, minXMaxY.y, maxXMaxY.x, maxXMaxY.y ) / divisor;
      }
    }

    return average;
  }

  /**
   * Returns a copy of the face that is clipped to be within the given axis-aligned bounding box.
   */
  public getClipped( minX: number, minY: number, maxX: number, maxY: number ): EdgedClippedFace {
    // TODO: consider whether containment checks are worth it. Most cases, no.
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

    this.forEachImplicitEdge( ( startPoint, endPoint ) => {
      PolygonClipping.boundsClipEdge(
        startPoint, endPoint,
        minX, minY, maxX, maxY, centerX, centerY,
        edges
      );
    } );

    // TODO: a more optimized form here! The clipping could output counts instead of us having to check here
    return EdgedClippedFace.fromEdges( edges, minX, minY, maxX, maxY );
  }

  /**
   * Returns two copies of the face, one that is clipped to be to the left of the given x value, and one that is
   * clipped to be to the right of the given x value.
   *
   * The fakeCornerY is used to determine the "fake" corner that is used for unsorted-edge clipping.
   */
  public getBinaryXClip( x: number, fakeCornerY: number ): { minFace: EdgedClippedFace; maxFace: EdgedClippedFace } {
    const minEdges: LinearEdge[] = [];
    const maxEdges: LinearEdge[] = [];

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      PolygonClipping.binaryXClipEdge( edge.startPoint, edge.endPoint, x, fakeCornerY, minEdges, maxEdges );
    }

    this.forEachImplicitEdge( ( startPoint, endPoint ) => {
      PolygonClipping.binaryXClipEdge( startPoint, endPoint, x, fakeCornerY, minEdges, maxEdges );
    } );

    assert && assert( minEdges.every( e => e.startPoint.x <= x && e.endPoint.x <= x ) );
    assert && assert( maxEdges.every( e => e.startPoint.x >= x && e.endPoint.x >= x ) );

    // TODO: a more optimized form here! The clipping could output counts instead of us having to check here
    return {
      minFace: EdgedClippedFace.fromEdges( minEdges, this.minX, this.minY, x, this.maxY ),
      maxFace: EdgedClippedFace.fromEdges( maxEdges, x, this.minY, this.maxX, this.maxY )
    };
  }

  /**
   * Returns two copies of the face, one that is clipped to y values less than the given y value, and one that is
   * clipped to values greater than the given y value.
   *
   * The fakeCornerX is used to determine the "fake" corner that is used for unsorted-edge clipping.
   */
  public getBinaryYClip( y: number, fakeCornerX: number ): { minFace: EdgedClippedFace; maxFace: EdgedClippedFace } {
    const minEdges: LinearEdge[] = [];
    const maxEdges: LinearEdge[] = [];

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      PolygonClipping.binaryYClipEdge( edge.startPoint, edge.endPoint, y, fakeCornerX, minEdges, maxEdges );
    }

    this.forEachImplicitEdge( ( startPoint, endPoint ) => {
      PolygonClipping.binaryYClipEdge( startPoint, endPoint, y, fakeCornerX, minEdges, maxEdges );
    } );

    assert && assert( minEdges.every( e => e.startPoint.y <= y && e.endPoint.y <= y ) );
    assert && assert( maxEdges.every( e => e.startPoint.y >= y && e.endPoint.y >= y ) );

    // TODO: a more optimized form here! The clipping could output counts instead of us having to check here
    return {
      minFace: EdgedClippedFace.fromEdges( minEdges, this.minX, this.minY, this.maxX, y ),
      maxFace: EdgedClippedFace.fromEdges( maxEdges, this.minX, y, this.maxX, this.maxY )
    };
  }

  /**
   * Returns two copies of the face, one that is clipped to contain points where dot( normal, point ) < value,
   * and one that is clipped to contain points where dot( normal, point ) > value.
   *
   * The fake corner perpendicular is used to determine the "fake" corner that is used for unsorted-edge clipping
   */
  public getBinaryLineClip( normal: Vector2, value: number, fakeCornerPerpendicular: number ): { minFace: EdgedClippedFace; maxFace: EdgedClippedFace } {
    const minEdges: LinearEdge[] = [];
    const maxEdges: LinearEdge[] = [];

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      PolygonClipping.binaryLineClipEdge( edge.startPoint, edge.endPoint, normal, value, fakeCornerPerpendicular, minEdges, maxEdges );
    }

    this.forEachImplicitEdge( ( startPoint, endPoint ) => {
      PolygonClipping.binaryLineClipEdge( startPoint, endPoint, normal, value, fakeCornerPerpendicular, minEdges, maxEdges );
    } );

    assert && assert( minEdges.every( e => normal.dot( e.startPoint ) <= value && normal.dot( e.endPoint ) <= value ) );
    assert && assert( maxEdges.every( e => normal.dot( e.startPoint ) >= value && normal.dot( e.endPoint ) >= value ) );

    // TODO: a more optimized form here! The clipping could output counts instead of us having to check here
    // NOTE: We can't really refine the bounds here.
    return {
      minFace: EdgedClippedFace.fromEdges( minEdges, this.minX, this.minY, this.maxX, this.maxY ),
      maxFace: EdgedClippedFace.fromEdges( maxEdges, this.minX, this.minY, this.maxX, this.maxY )
    };
  }

  /**
   * Returns an array of faces, clipped similarly to getBinaryLineClip, but with more than one (parallel) split line at
   * a time. The first face will be the one with dot( normal, point ) < values[0], the second one with
   * values[ 0 ] < dot( normal, point ) < values[1], etc.
   */
  public getStripeLineClip( normal: Vector2, values: number[], fakeCornerPerpendicular: number ): EdgedClippedFace[] {
    if ( values.length === 0 ) {
      return [ this ];
    }

    const edgesCollection: LinearEdge[][] = _.range( values.length + 1 ).map( () => [] );

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      PolygonClipping.binaryStripeClipEdge( edge.startPoint, edge.endPoint, normal, values, fakeCornerPerpendicular, edgesCollection );
    }

    this.forEachImplicitEdge( ( startPoint, endPoint ) => {
      PolygonClipping.binaryStripeClipEdge( startPoint, endPoint, normal, values, fakeCornerPerpendicular, edgesCollection );
    } );

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

    return edgesCollection.map( edges => EdgedClippedFace.fromEdges( edges, this.minX, this.minY, this.maxX, this.maxY ) );
  }

  /**
   * Returns two copies of the face, one that is clipped to contain points inside the circle defined by the given
   * center and radius, and one that is clipped to contain points outside the circle.
   *
   * NOTE: maxAngleSplit is used to determine the polygonal approximation of the circle. The returned result will not
   * have a chord with an angle greater than maxAngleSplit.
   */
  public getBinaryCircularClip( center: Vector2, radius: number, maxAngleSplit: number ): { insideFace: EdgedClippedFace; outsideFace: EdgedClippedFace } {
    const insideEdges: LinearEdge[] = [];
    const outsideEdges: LinearEdge[] = [];

    PolygonClipping.binaryCircularClipEdges( this.getAllEdges(), center, radius, maxAngleSplit, insideEdges, outsideEdges );

    return {
      insideFace: EdgedClippedFace.fromEdges( insideEdges, this.minX, this.minY, this.maxX, this.maxY ),
      outsideFace: EdgedClippedFace.fromEdges( outsideEdges, this.minX, this.minY, this.maxX, this.maxY )
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

    this.forEachImplicitEdge( ( startPoint, endPoint ) => {
      PolygonClipping.gridClipIterate(
        startPoint, endPoint,
        minX, minY, maxX, maxY,
        stepX, stepY, stepWidth, stepHeight,
        callback
      );
    } );

    if ( this.edges.length ) {
      polygonCompleteCallback();
    }
  }

  /**
   * Returns the evaluation of the bilinear (tent) filter integrals for the given point, ASSUMING that the face
   * is clipped to the transformed unit square of x: [minX,minX+1], y: [minY,minY+1].
   */
  public getBilinearFiltered( pointX: number, pointY: number, minX: number, minY: number ): number {
    // TODO: optimization
    // TODO: we REALLY should have a ClippedFace primitive?
    return PolygonBilinear.evaluateLinearEdges( this.getAllEdges(), pointX, pointY, minX, minY );
  }

  /**
   * Returns the evaluation of the Mitchell-Netravali (1/3,1/3) filter integrals for the given point, ASSUMING that the
   * face is clipped to the transformed unit square of x: [minX,minX+1], y: [minY,minY+1].
   */
  public getMitchellNetravaliFiltered( pointX: number, pointY: number, minX: number, minY: number ): number {
    // TODO: optimization
    return PolygonMitchellNetravali.evaluateLinearEdges( this.getAllEdges(), pointX, pointY, minX, minY );
  }

  /**
   * Returns whether the face contains the given point.
   */
  public containsPoint( point: Vector2 ): boolean {
    let windingNumber = LinearEdge.getWindingNumberEdges( this.edges, point );

    this.forEachImplicitEdge( ( startPoint, endPoint ) => {
      windingNumber += LinearEdge.windingContribution(
        startPoint.x, startPoint.y, endPoint.x, endPoint.y, point.x, point.y
      );
    } );

    return windingNumber !== 0;
  }

  /**
   * Returns an affine-transformed version of the face.
   */
  public getTransformed( transform: Matrix3 ): EdgedClippedFace {
    if ( transform.isIdentity() ) {
      return this;
    }
    else {
      const transformedEdges: LinearEdge[] = [];

      const allEdges = this.getAllEdges();

      for ( let i = 0; i < allEdges.length; i++ ) {
        const edge = allEdges[ i ];

        const start = transform.timesVector2( edge.startPoint );
        const end = transform.timesVector2( edge.endPoint );

        if ( !start.equals( end ) ) {
          transformedEdges.push( new LinearEdge( start, end ) );
        }
      }

      const bounds = new Bounds2( this.minX, this.minY, this.maxX, this.maxY ).transform( transform );

      return EdgedClippedFace.fromEdgesWithNoCheck( transformedEdges, bounds.minX, bounds.minY, bounds.maxX, bounds.maxY );
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

    return EdgedClippedFace.fromEdgesWithNoCheck(
      edges,
      Utils.roundSymmetric( this.minX / epsilon ) * epsilon,
      Utils.roundSymmetric( this.minY / epsilon ) * epsilon,
      Utils.roundSymmetric( this.maxX / epsilon ) * epsilon,
      Utils.roundSymmetric( this.maxY / epsilon ) * epsilon
    );
  }

  /**
   * Calls the callback with points for each edge in the face.
   */
  public forEachEdge( callback: ( startPoint: Vector2, endPoint: Vector2 ) => void ): void {
    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];
      callback( edge.startPoint, edge.endPoint );
    }

    this.forEachImplicitEdge( callback );
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
    return new EdgedClippedFaceAccumulator();
  }

  /**
   * Returns a debugging string.
   */
  public toString(): string {
    return this.getAllEdges().map( e => `${e.startPoint.x},${e.startPoint.y} => ${e.endPoint.x},${e.endPoint.y}` ).join( '\n' );
  }

  /**
   * Returns a serialized version of the face, that should be able to be deserialized into the same type of face.
   * See {FaceType}.deserialize.
   *
   * NOTE: If you don't know what type of face this is, use serializeClippableFace instead.
   */
  public serialize(): SerializedEdgedClippedFace {
    return {
      edges: this.edges.map( edge => edge.serialize() ),
      minX: this.minX,
      minY: this.minY,
      maxX: this.maxX,
      maxY: this.maxY,
      minXCount: this.minXCount,
      minYCount: this.minYCount,
      maxXCount: this.maxXCount,
      maxYCount: this.maxYCount
    };
  }

  public static deserialize( serialized: SerializedEdgedClippedFace ): EdgedFace {
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

scenery.register( 'EdgedClippedFace', EdgedClippedFace );

export class EdgedClippedFaceAccumulator implements ClippableFaceAccumulator {

  private edges: LinearEdge[] = [];
  private minX = 0;
  private minY = 0;
  private maxX = 0;
  private maxY = 0;
  private minXCount = 0;
  private minYCount = 0;
  private maxXCount = 0;
  private maxYCount = 0;

  public addEdge( startX: number, startY: number, endX: number, endY: number, startPoint: Vector2 | null, endPoint: Vector2 | null ): void {
    if (
      // If all points are on a corner
      startX === this.minX || startX === this.maxX &&
      startY === this.minY || startY === this.maxY &&
      endX === this.minX || endX === this.maxX &&
      endY === this.minY || endY === this.maxY &&
      // And we're not on opposite corners
      startX === endX || startY === endY
    ) {
      assert && assert( startX !== endX || startY !== endY, 'Points should not be identical' );

      if ( startX === endX ) {
        const delta = ( startY === this.minY ? 1 : -1 );
        if ( startX === this.minX ) {
          this.minXCount += delta;
        }
        else {
          this.maxXCount += delta;
        }
      }
      else {
        const delta = ( startX === this.minX ? 1 : -1 );
        if ( startY === this.minY ) {
          this.minYCount += delta;
        }
        else {
          this.maxYCount += delta;
        }
      }
    }
    else {
      this.edges.push( new LinearEdge(
        startPoint || new Vector2( startX, startY ),
        endPoint || new Vector2( endX, endY )
      ) );
    }
  }

  public markNewPolygon(): void {
    // no-op, since we're storing unsorted edges!
  }

  public setAccumulationBounds( minX: number, minY: number, maxX: number, maxY: number ): void {
    this.minX = minX;
    this.minY = minY;
    this.maxX = maxX;
    this.maxY = maxY;
    this.minXCount = 0;
    this.minYCount = 0;
    this.maxXCount = 0;
    this.maxYCount = 0;
  }

  // Will reset it to the initial state also
  public finalizeFace(): EdgedClippedFace | null {
    if ( this.edges.length === 0 ) {
      return null;
    }

    const edges = this.edges;
    this.edges = [];
    return new EdgedClippedFace( edges, this.minX, this.minY, this.maxX, this.maxY, this.minXCount, this.minYCount, this.maxXCount, this.maxYCount );
  }

  public finalizeEnsureFace( minX: number, minY: number, maxX: number, maxY: number ): EdgedClippedFace {
    return this.finalizeFace() || new EdgedClippedFace( emptyArray, minX, minY, maxX, maxY, 0, 0, 0, 0 );
  }

  // Will reset without creating a face
  public reset(): void {
    this.edges.length = 0;
    this.minXCount = 0;
    this.minYCount = 0;
    this.maxXCount = 0;
    this.maxYCount = 0;
  }
}

const scratchAccumulator = new EdgedClippedFaceAccumulator();

export type SerializedEdgedClippedFace = {
  edges: SerializedLinearEdge[];
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  minXCount: number;
  minYCount: number;
  maxXCount: number;
  maxYCount: number;
};
