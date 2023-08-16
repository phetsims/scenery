// Copyright 2023, University of Colorado Boulder

/**
 * A line segment (between two points).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

export default class LinearEdge {

  // NOTE: We'll flag these, so that we can accurately compute bounds later when desired (and can skip edges with
  // corner vertices).
  // TODO: how to handle this for performance?

  public constructor(
    public readonly startPoint: Vector2,
    public readonly endPoint: Vector2,
    public readonly containsFakeCorner: boolean = false // TODO: propagate fake corners
  ) {
    assert && assert( !startPoint.equals( endPoint ) );
  }

  public static fromPolygon( polygon: Vector2[] ): LinearEdge[] {
    const edges: LinearEdge[] = [];

    for ( let i = 0; i < polygon.length; i++ ) {
      edges.push( new LinearEdge(
        polygon[ i ],
        polygon[ ( i + 1 ) % polygon.length ]
      ) );
    }

    return edges;
  }

  public static fromPolygons( polygons: Vector2[][] ): LinearEdge[] {
    return polygons.flatMap( LinearEdge.fromPolygon );
  }

  // TODO: ideally a better version of this?
  public static toPolygons( edges: LinearEdge[] ): Vector2[][] {
    const polygons: Vector2[][] = [];

    const remainingEdges = new Set<LinearEdge>( edges );

    while ( remainingEdges.size > 0 ) {
      const edge: LinearEdge = remainingEdges.values().next().value;

      const polygon: Vector2[] = [];

      let currentEdge = edge;
      do {
        polygon.push( currentEdge.startPoint );
        remainingEdges.delete( currentEdge );
        currentEdge = [ ...remainingEdges ].find( e => e.startPoint.equalsEpsilon( currentEdge.endPoint, 1e-8 ) )!; // eslint-disable-line @typescript-eslint/no-loop-func
      } while ( currentEdge !== edge );

      assert && assert( polygon.length >= 3 );

      polygons.push( polygon );
    }

    return polygons;
  }

  // Cancelled subexpressions for fewer multiplications
  public static evaluateLineIntegralShoelaceArea( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    return 0.5 * ( p1x + p0x ) * ( p1y - p0y );
  }

  // Without the subexpression cancelling
  public static evaluateLineIntegralArea( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    return 0.5 * ( p0x * p1y - p0y * p1x );
  }

  /**
   * If you take the sum of these for a closed polygon and DIVIDE IT by the area, it should be the centroid of the
   * polygon.
   */
  public static evaluateLineIntegralPartialCentroid( p0x: number, p0y: number, p1x: number, p1y: number ): Vector2 {
    const base = ( 1 / 6 ) * ( p0x * p1y - p1x * p0y );

    return new Vector2(
      ( p0x + p1x ) * base,
      ( p0y + p1y ) * base
    );
  }

  public static evaluateLineIntegralZero( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    return ( p0x - 0.1396 ) * ( p0y + 1.422 ) - ( p1x - 0.1396 ) * ( p1y + 1.422 );
  }

  public static evaluateLineIntegralDistance( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    const dx = p1x - p0x;
    const dy = p1y - p0y;
    const a = p0x * p1y - p0y * p1x;
    const qd = Math.sqrt( dx * dx + dy * dy );
    const q0 = Math.sqrt( p0x * p0x + p0y * p0y );
    const q1 = Math.sqrt( p1x * p1x + p1y * p1y );
    const kx = p1x * p1x - p0x * p1x;
    const ky = p1y * p1y - p0y * p1y;

    return a / ( 6 * qd * qd * qd ) * (
      qd * ( q0 * ( p0x * p0x - p0x * p1x - p0y * dy ) + q1 * ( kx + p1y * dy ) ) +
      a * a * ( Math.log( ( kx + ky + qd * q1 ) / ( p0x * dx + q0 * qd + p0y * dy ) ) )
    );
  }

  public static evaluateCancelledX( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    return ( 1 / 6 ) * ( p0x + p1x ) * ( p0x * p1y - p1x * p0y );
  }

  public static evaluateCancelledY( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    return ( 1 / 6 ) * ( p0y + p1y ) * ( p0x * p1y - p1x * p0y );
  }

  /**
   * If you take the sum of these for a closed polygon, it should be the area of the polygon.
   */
  public getLineIntegralArea(): number {
    return LinearEdge.evaluateLineIntegralShoelaceArea(
      this.startPoint.x, this.startPoint.y, this.endPoint.x, this.endPoint.y
    );
  }

  public getLineIntegralPartialCentroid(): Vector2 {
    return LinearEdge.evaluateLineIntegralPartialCentroid(
      this.startPoint.x, this.startPoint.y, this.endPoint.x, this.endPoint.y
    );
  }

  public getLineIntegralDistance(): number {
    return LinearEdge.evaluateLineIntegralDistance(
      this.startPoint.x, this.startPoint.y, this.endPoint.x, this.endPoint.y
    );
  }

  // TODO: use this to check all of our LinearEdge computations
  /**
   * If you take the sum of these for a closed polygon, it should be zero (used to check computations).
   */
  public getLineIntegralZero(): number {
    return LinearEdge.evaluateLineIntegralZero(
      this.startPoint.x, this.startPoint.y, this.endPoint.x, this.endPoint.y
    );
  }

  public static getPolygonArea( polygon: Vector2[] ): number {
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

  public static getPolygonCentroid( polygon: Vector2[] ): Vector2 {
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

    const area = LinearEdge.getPolygonArea( polygon );

    return new Vector2(
      x / area,
      y / area
    );
  }

  public static getEdgesArea( clippedEdges: LinearEdge[] ): number {
    let sum = 0;

    for ( let i = 0; i < clippedEdges.length; i++ ) {
      sum += clippedEdges[ i ].getLineIntegralArea();
    }

    return sum;
  }
}

scenery.register( 'LinearEdge', LinearEdge );
