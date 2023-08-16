// Copyright 2023, University of Colorado Boulder

/**
 * Test rasterization
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, PolygonClipping, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Range from '../../../../../dot/js/Range.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

// Relies on the main boundary being positive-oriented, and the holes being negative-oriented and non-overlapping
export default class PolygonalFace implements ClippableFace {
  public constructor( public readonly polygons: Vector2[][] ) {}

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
}

scenery.register( 'PolygonalFace', PolygonalFace );
