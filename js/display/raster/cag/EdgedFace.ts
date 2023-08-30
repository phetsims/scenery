// Copyright 2023, University of Colorado Boulder

/**
 * A ClippableFace from a set of line segment edges. Should still represent multiple closed loops, but it is not
 * explicit.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, LinearEdge, PolygonalFace, PolygonBilinear, PolygonClipping, PolygonMitchellNetravali, scenery, SerializedLinearEdge } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Range from '../../../../../dot/js/Range.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Utils from '../../../../../dot/js/Utils.js';
import { Shape } from '../../../../../kite/js/imports.js';

const scratchVectorA = new Vector2( 0, 0 );
const scratchVectorB = new Vector2( 0, 0 );

export default class EdgedFace implements ClippableFace {
  public constructor( public readonly edges: LinearEdge[] ) {}

  public toPolygonalFace( epsilon = 1e-8 ): PolygonalFace {
    return new PolygonalFace( LinearEdge.toPolygons( this.edges, epsilon ) );
  }

  public toEdgedFace(): EdgedFace {
    return this;
  }

  public getShape( epsilon = 1e-8 ): Shape {
    return this.toPolygonalFace( epsilon ).getShape();
  }

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

  public getDistanceRange( point: Vector2 ): Range {
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

  public getArea(): number {
    let area = 0;
    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      const p0 = edge.startPoint;
      const p1 = edge.endPoint;
      // PolygonIntegrals.evaluateShoelaceArea( p0.x, p0.y, p1.x, p1.y );
      area += ( p1.x + p0.x ) * ( p1.y - p0.y );
    }

    return 0.5 * area;
  }

  public getCentroidPartial(): Vector2 {
    let x = 0;
    let y = 0;

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];

      const p0 = edge.startPoint;
      const p1 = edge.endPoint;

      // evaluateCentroidPartial
      const base = ( 1 / 6 ) * ( p0.x * p1.y - p1.x * p0.y );
      x += ( p0.x + p1.x ) * base;
      y += ( p0.y + p1.y ) * base;
    }

    return new Vector2( x, y );
  }

  public getCentroid( area: number ): Vector2 {
    return this.getCentroidPartial().timesScalar( 1 / area );
  }

  public getZero(): number {
    return _.sum( this.edges.map( e => e.getLineIntegralZero() ) );
  }

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

  public getClipped( bounds: Bounds2 ): EdgedFace {
    const edges: LinearEdge[] = [];

    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];
      PolygonClipping.boundsClipEdge( edge.startPoint, edge.endPoint, bounds, edges );
    }

    return new EdgedFace( edges );
  }

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

  public getBinaryCircularClip( center: Vector2, radius: number, maxAngleSplit: number ): { insideFace: EdgedFace; outsideFace: EdgedFace } {
    const insideEdges: LinearEdge[] = [];
    const outsideEdges: LinearEdge[] = [];

    PolygonClipping.binaryCircularClipEdges( this.edges, center, radius, maxAngleSplit, insideEdges, outsideEdges );

    return {
      insideFace: new EdgedFace( insideEdges ),
      outsideFace: new EdgedFace( outsideEdges )
    };
  }

  public getBilinearFiltered( pointX: number, pointY: number, minX: number, minY: number ): number {
    return PolygonBilinear.evaluateLinearEdges( this.edges, pointX, pointY, minX, minY );
  }

  public getMitchellNetravaliFiltered( pointX: number, pointY: number, minX: number, minY: number ): number {
    return PolygonMitchellNetravali.evaluateLinearEdges( this.edges, pointX, pointY, minX, minY );
  }

  public containsPoint( point: Vector2 ): boolean {
    return LinearEdge.getWindingNumberEdges( this.edges, point ) !== 0;
  }

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

  public getRounded( epsilon: number ): EdgedFace {
    return new EdgedFace( this.edges.map( edge => new LinearEdge(
      new Vector2(
        Utils.roundSymmetric( edge.startPoint.x / epsilon ) * epsilon,
        Utils.roundSymmetric( edge.startPoint.y / epsilon ) * epsilon
      ),
      new Vector2(
        Utils.roundSymmetric( edge.endPoint.x / epsilon ) * epsilon,
        Utils.roundSymmetric( edge.endPoint.y / epsilon ) * epsilon
      ),
      edge.containsFakeCorner
    ) ) );
  }

  public forEachEdge( callback: ( startPoint: Vector2, endPoint: Vector2 ) => void ): void {
    for ( let i = 0; i < this.edges.length; i++ ) {
      const edge = this.edges[ i ];
      callback( edge.startPoint, edge.endPoint );
    }
  }

  public toString(): string {
    return this.edges.map( e => `${e.startPoint.x},${e.startPoint.y} => ${e.endPoint.x},${e.endPoint.y}` ).join( '\n' );
  }

  public serialize(): SerializedEdgedFace {
    return {
      edges: this.edges.map( edge => edge.serialize() )
    };
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

export type SerializedEdgedFace = {
  edges: SerializedLinearEdge[];
};
