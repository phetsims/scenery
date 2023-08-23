// Copyright 2023, University of Colorado Boulder

/**
 * Utilities for the bilinear/tent filter for polygons
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import { LinearEdge, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

export default class PolygonBilinear {
  public static evaluate( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    const c01 = p0x * p1y;
    const c10 = p1x * p0y;
    return ( c01 - c10 ) * ( 12 - 4 * ( p0x + p0y + p1x + p1y ) + 2 * ( p0x * p0y + p1x * p1y ) + c10 + c01 ) / 24;
  }

  private static getSign( pointX: number, pointY: number, minX: number, minY: number ): number {
    const offsetX = minX - pointX;
    const offsetY = minY - pointY;

    assert && assert( offsetX === -1 || offsetX === 0 );
    assert && assert( offsetY === -1 || offsetY === 0 );

    return ( offsetX === offsetY ) ? 1 : -1;
  }

  public static evaluatePolygons( polygons: Vector2[][], pointX: number, pointY: number, minX: number, minY: number ): number {
    let sum = 0;

    for ( let i = 0; i < polygons.length; i++ ) {
      const polygon = polygons[ i ];

      const lastPoint = polygon[ polygon.length - 1 ];
      let lastX = Math.abs( lastPoint.x - pointX );
      let lastY = Math.abs( lastPoint.y - pointY );
      for ( let j = 0; j < polygon.length; j++ ) {
        const point = polygon[ j ];
        const x = Math.abs( point.x - pointX );
        const y = Math.abs( point.y - pointY );

        sum += PolygonBilinear.evaluate( lastX, lastY, x, y );

        lastX = x;
        lastY = y;
      }
    }

    return sum * PolygonBilinear.getSign( pointX, pointY, minX, minY );
  }

  public static evaluateLinearEdges( edges: LinearEdge[], pointX: number, pointY: number, minX: number, minY: number ): number {
    let sum = 0;

    for ( let i = 0; i < edges.length; i++ ) {
      const edge = edges[ i ];

      const p0x = Math.abs( edge.startPoint.x - pointX );
      const p0y = Math.abs( edge.startPoint.y - pointY );
      const p1x = Math.abs( edge.endPoint.x - pointX );
      const p1y = Math.abs( edge.endPoint.y - pointY );

      sum += PolygonBilinear.evaluate( p0x, p0y, p1x, p1y );
    }

    return sum * PolygonBilinear.getSign( pointX, pointY, minX, minY );
  }
}

scenery.register( 'PolygonBilinear', PolygonBilinear );
