// Copyright 2023, University of Colorado Boulder

/**
 * Assorted useful integrals for polygons
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import { scenery } from '../../imports.js';
import Vector2 from '../../../../dot/js/Vector2.js';

export default class PolygonIntegrals {
  public static getArea( polygon: Vector2[] ): number {
    let sum = 0;

    // TODO: micro-optimize if used?
    for ( let i = 0; i < polygon.length; i++ ) {
      const p0 = polygon[ i % polygon.length ];
      const p1 = polygon[ ( i + 1 ) % polygon.length ];

      // PolygonIntegrals.evaluateShoelaceArea( p0.x, p0.y, p1.x, p1.y );
      sum += 0.5 * ( p1.x + p0.x ) * ( p1.y - p0.y );
    }

    return sum;
  }

  public static getCentroid( polygon: Vector2[] ): Vector2 {
    let x = 0;
    let y = 0;

    // TODO: micro-optimize if used?
    for ( let i = 0; i < polygon.length; i++ ) {
      const p0 = polygon[ i % polygon.length ];
      const p1 = polygon[ ( i + 1 ) % polygon.length ];

      // evaluateCentroidPartial
      const base = ( 1 / 6 ) * ( p0.x * p1.y - p1.x * p0.y );
      x += ( p0.x + p1.x ) * base;
      y += ( p0.y + p1.y ) * base;
    }

    const area = PolygonIntegrals.getArea( polygon );

    return new Vector2(
      x / area,
      y / area
    );
  }

  public static evaluateShoelaceArea( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    return 0.5 * ( p1x + p0x ) * ( p1y - p0y );
  }

  public static evaluateCentroidPartial( p0x: number, p0y: number, p1x: number, p1y: number ): Vector2 {
    const base = ( 1 / 6 ) * ( p0x * p1y - p1x * p0y );

    return new Vector2(
      ( p0x + p1x ) * base,
      ( p0y + p1y ) * base
    );
  }

  public static evaluateCancelledX( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    return ( 1 / 6 ) * ( p0x + p1x ) * ( p0x * p1y - p1x * p0y );
  }

  public static evaluateCancelledY( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    return ( 1 / 6 ) * ( p0y + p1y ) * ( p0x * p1y - p1x * p0y );
  }
}

scenery.register( 'PolygonIntegrals', PolygonIntegrals );
