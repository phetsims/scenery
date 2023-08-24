// Copyright 2023, University of Colorado Boulder

/**
 * A ClippableFace from a set of polygons (each one is a closed loop of Vector2s)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, EdgedFace, LinearEdge, PolygonBilinear, PolygonClipping, PolygonMitchellNetravali, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Range from '../../../../../dot/js/Range.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';
import Utils from '../../../../../dot/js/Utils.js';
import { Shape } from '../../../../../kite/js/imports.js';

const scratchVectorA = new Vector2( 0, 0 );
const scratchVectorB = new Vector2( 0, 0 );

// Relies on the main boundary being positive-oriented, and the holes being negative-oriented and non-overlapping
export default class PolygonalFace implements ClippableFace {
  public constructor( public readonly polygons: Vector2[][] ) {}

  public toEdgedFace(): EdgedFace {
    return new EdgedFace( LinearEdge.fromPolygons( this.polygons ) );
  }

  public toPolygonalFace( epsilon?: number ): PolygonalFace {
    return this;
  }

  public getShape( epsilon?: number ): Shape {
    return LinearEdge.polygonsToShape( this.polygons );
  }

  public getBounds(): Bounds2 {
    const result = Bounds2.NOTHING.copy();
    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];
      for ( let j = 0; j < polygon.length; j++ ) {
        result.addPoint( polygon[ j ] );
      }
    }
    return result;
  }

  public getDotRange( normal: Vector2 ): Range {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;

    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];
      for ( let j = 0; j < polygon.length; j++ ) {
        const dot = polygon[ j ].dot( normal );
        min = Math.min( min, dot );
        max = Math.max( max, dot );
      }
    }

    return new Range( min, max );
  }

  public getDistanceRange( point: Vector2 ): Range {
    let min = Number.POSITIVE_INFINITY;
    let max = 0;

    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];
      for ( let j = 0; j < polygon.length; j++ ) {
        const p0 = polygon[ j ];
        const p1 = polygon[ ( j + 1 ) % polygon.length ];

        const p0x = p0.x - point.x;
        const p0y = p0.y - point.y;
        const p1x = p1.x - point.x;
        const p1y = p1.y - point.y;

        min = Math.min( min, LinearEdge.evaluateClosestDistanceToOrigin( p0x, p0y, p1x, p1y ) );
        max = Math.max( max, Math.sqrt( p0x * p0x + p0y * p0y ), Math.sqrt( p1x * p1x + p1y * p1y ) );
      }
    }

    return new Range( min, max );
  }

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

  public getAverageDistance( point: Vector2, area: number ): number {
    let sum = 0;

    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];

      // TODO: optimize more?
      for ( let j = 0; j < polygon.length; j++ ) {
        const p0 = polygon[ j ];
        const p1 = polygon[ ( j + 1 ) % polygon.length ];

      sum += LinearEdge.evaluateLineIntegralDistance(
        p0.x - point.x,
        p0.y - point.y,
        p1.x - point.x,
        p1.y - point.y
      );
      }
    }

    return sum / area;
  }

  public getAverageDistanceTransformedToOrigin( transform: Matrix3, area: number ): number {
    let sum = 0;

    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];

      // TODO: optimize more? THIS WILL BE TRICKY due to not creating garbage. rotate scratch vectors!
      for ( let j = 0; j < polygon.length; j++ ) {
        const p0 = transform.multiplyVector2( scratchVectorA.set( polygon[ j ] ) );
        const p1 = transform.multiplyVector2( scratchVectorB.set( polygon[ ( j + 1 ) % polygon.length ] ) );

        sum += LinearEdge.evaluateLineIntegralDistance( p0.x, p0.y, p1.x, p1.y );
      }
    }

    return sum / ( area * transform.getSignedScale() );
  }

  public getClipped( bounds: Bounds2 ): PolygonalFace {
    return new PolygonalFace( this.polygons.map( polygon => PolygonClipping.boundsClipPolygon( polygon, bounds ) ) );
  }

  public getBinaryXClip( x: number, fakeCornerY: number ): { minFace: PolygonalFace; maxFace: PolygonalFace } {
    const minPolygons: Vector2[][] = [];
    const maxPolygons: Vector2[][] = [];

    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];

      const minPolygon: Vector2[] = [];
      const maxPolygon: Vector2[] = [];

      PolygonClipping.binaryXClipPolygon( polygon, x, minPolygon, maxPolygon );

      minPolygon.length && minPolygons.push( minPolygon );
      maxPolygon.length && maxPolygons.push( maxPolygon );

      assert && assert( minPolygon.every( p => p.x <= x ) );
      assert && assert( maxPolygon.every( p => p.x >= x ) );
    }

    return {
      minFace: new PolygonalFace( minPolygons ),
      maxFace: new PolygonalFace( maxPolygons )
    };
  }

  public getBinaryYClip( y: number, fakeCornerX: number ): { minFace: PolygonalFace; maxFace: PolygonalFace } {
    const minPolygons: Vector2[][] = [];
    const maxPolygons: Vector2[][] = [];

    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];

      const minPolygon: Vector2[] = [];
      const maxPolygon: Vector2[] = [];

      PolygonClipping.binaryYClipPolygon( polygon, y, minPolygon, maxPolygon );

      minPolygon.length && minPolygons.push( minPolygon );
      maxPolygon.length && maxPolygons.push( maxPolygon );

      assert && assert( minPolygon.every( p => p.y <= y ) );
      assert && assert( maxPolygon.every( p => p.y >= y ) );
    }

    return {
      minFace: new PolygonalFace( minPolygons ),
      maxFace: new PolygonalFace( maxPolygons )
    };
  }

  public getBinaryLineClip( normal: Vector2, value: number, fakeCornerPerpendicular: number ): { minFace: PolygonalFace; maxFace: PolygonalFace } {
    const minPolygons: Vector2[][] = [];
    const maxPolygons: Vector2[][] = [];

    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];

      const minPolygon: Vector2[] = [];
      const maxPolygon: Vector2[] = [];

      PolygonClipping.binaryLineClipPolygon( polygon, normal, value, minPolygon, maxPolygon );

      minPolygon.length && minPolygons.push( minPolygon );
      maxPolygon.length && maxPolygons.push( maxPolygon );

      assert && assert( minPolygon.every( p => normal.dot( p ) - 1e-8 <= value ) );
      assert && assert( maxPolygon.every( p => normal.dot( p ) + 1e-8 >= value ) );
    }

    return {
      minFace: new PolygonalFace( minPolygons ),
      maxFace: new PolygonalFace( maxPolygons )
    };
  }

  public getStripeLineClip( normal: Vector2, values: number[], fakeCornerPerpendicular: number ): PolygonalFace[] {
    const polygonsCollection: Vector2[][][] = _.range( values.length + 1 ).map( () => [] );

    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];

      const polygons = PolygonClipping.binaryStripeClipPolygon( polygon, normal, values );

      assert && assert( polygonsCollection.length === polygons.length );

      for ( let j = 0; j < polygons.length; j++ ) {
        const slicePolygon = polygons[ j ];

        polygonsCollection[ j ].push( slicePolygon );

        if ( assert ) {
          const minValue = j > 0 ? values[ j - 1 ] : Number.NEGATIVE_INFINITY;
          const maxValue = j < values.length ? values[ j ] : Number.POSITIVE_INFINITY;

          assert( slicePolygon.every( p => normal.dot( p ) + 1e-8 >= minValue && normal.dot( p ) - 1e-8 <= maxValue ) );
        }
      }
    }

    return polygonsCollection.map( polygons => new PolygonalFace( polygons ) );
  }

  // NOTE: switches to EdgedFaces! Could probably implement the binary circular clip for polygons, but it seems a bit
  // harder
  public getBinaryCircularClip( center: Vector2, radius: number, maxAngleSplit: number ): { insideFace: PolygonalFace; outsideFace: PolygonalFace } {

    // TODO: This can be deleted after we've fully tested the new clipping. It's a good 450 lines of complicated stuff
    // return this.toEdgedFace().getBinaryCircularClip( center, radius, maxAngleSplit );

    const insidePolygons: Vector2[][] = [];
    const outsidePolygons: Vector2[][] = [];

    PolygonClipping.binaryCircularClipPolygon( this.polygons, center, radius, maxAngleSplit, insidePolygons, outsidePolygons );

    return {
      insideFace: new PolygonalFace( insidePolygons ),
      outsideFace: new PolygonalFace( outsidePolygons )
    };
  }

  public getBilinearFiltered( pointX: number, pointY: number, minX: number, minY: number ): number {
    return PolygonBilinear.evaluatePolygons( this.polygons, pointX, pointY, minX, minY );
  }

  public getMitchellNetravaliFiltered( pointX: number, pointY: number, minX: number, minY: number ): number {
    return PolygonMitchellNetravali.evaluatePolygons( this.polygons, pointX, pointY, minX, minY );
  }

  public getTransformed( transform: Matrix3 ): PolygonalFace {
    if ( transform.isIdentity() ) {
      return this;
    }
    else {
      return new PolygonalFace( this.polygons.map( polygon => polygon.map( vertex => {
        return transform.timesVector2( vertex );
      } ) ) );
    }
  }

  public getRounded( epsilon: number ): PolygonalFace {
    return new PolygonalFace( this.polygons.map( polygon => polygon.map( vertex => {
      return new Vector2(
        Utils.roundSymmetric( vertex.x / epsilon ) * epsilon,
        Utils.roundSymmetric( vertex.y / epsilon ) * epsilon
      );
    } ) ) );
  }

  public forEachEdge( callback: ( startPoint: Vector2, endPoint: Vector2 ) => void ): void {
    for ( let i = 0; i < this.polygons.length; i++ ) {
      const polygon = this.polygons[ i ];
      for ( let j = 0; j < polygon.length; j++ ) {
        callback( polygon[ j ], polygon[ ( j + 1 ) % polygon.length ] );
      }
    }
  }

  public toString(): string {
    return this.polygons.map( polygon => polygon.map( p => `${p.x},${p.y}` ).join( ' ' ) ).join( '\n' );
  }

  public serialize(): SerializedPolygonalFace {
    return {
      polygons: this.polygons.map( polygon => polygon.map( p => ( { x: p.x, y: p.y } ) ) )
    };
  }

  public static deserialize( serialized: SerializedPolygonalFace ): PolygonalFace {
    return new PolygonalFace( serialized.polygons.map( polygon => polygon.map( p => new Vector2( p.x, p.y ) ) ) );
  }

  public static fromBounds( bounds: Bounds2 ): PolygonalFace {
    return PolygonalFace.fromBoundsValues( bounds.minX, bounds.minY, bounds.maxX, bounds.maxY );
  }

  public static fromBoundsValues( minX: number, minY: number, maxX: number, maxY: number ): PolygonalFace {
    return new PolygonalFace( [ [
      new Vector2( minX, minY ),
      new Vector2( maxX, minY ),
      new Vector2( maxX, maxY ),
      new Vector2( minX, maxY )
    ] ] );
  }
}

scenery.register( 'PolygonalFace', PolygonalFace );

export type SerializedPolygonalFace = {
  polygons: { x: number; y: number }[][];
};
