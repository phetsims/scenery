// Copyright 2023, University of Colorado Boulder

/**
 * A line segment (between two points).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClipSimplifier, scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import { Shape } from '../../../../../kite/js/imports.js';

export default class LinearEdge {

  // NOTE: We'll flag these, so that we can accurately compute bounds later when desired (and can skip edges with
  // corner vertices).
  // TODO: how to handle this for performance?

  public constructor(
    public readonly startPoint: Vector2,
    public readonly endPoint: Vector2,
    public readonly containsFakeCorner: boolean = false // TODO: propagate fake corners
  ) {
    assert && assert( startPoint.isFinite() );
    assert && assert( endPoint.isFinite() );
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
  public static toPolygons( edges: LinearEdge[], epsilon = 1e-8 ): Vector2[][] {
    const filteredEdges = LinearEdge.withOppositesRemoved( edges );

    const polygons: Vector2[][] = [];

    const remainingEdges = new Set<LinearEdge>( filteredEdges );

    while ( remainingEdges.size > 0 ) {
      const edge: LinearEdge = remainingEdges.values().next().value;

      const simplifier = new ClipSimplifier( true );

      let currentEdge = edge;
      do {
        simplifier.add( currentEdge.startPoint.x, currentEdge.startPoint.y );
        remainingEdges.delete( currentEdge );
        if ( edge.startPoint.equalsEpsilon( currentEdge.endPoint, epsilon ) ) {
          break;
        }
        else {
          currentEdge = [ ...remainingEdges ].find( candidateEdge => { // eslint-disable-line @typescript-eslint/no-loop-func
            return candidateEdge.startPoint.equalsEpsilon( currentEdge.endPoint, epsilon );
          } )!;
        }
      } while ( currentEdge !== edge );

      const polygon = simplifier.finalize();

      if ( polygon.length >= 3 ) {
        polygons.push( polygon );
      }
    }

    return polygons;
  }

  // TODO: can we go with a stronger form also, that finds everything collinear, and simplifies it?
  public static withOppositesRemoved( edges: LinearEdge[], epsilon = 1e-8 ): LinearEdge[] {
    const outputEdges = [];
    const remainingEdges = new Set<LinearEdge>( edges );

    while ( remainingEdges.size > 0 ) {
      const edge: LinearEdge = remainingEdges.values().next().value;
      remainingEdges.delete( edge );

      const opposite = [ ...remainingEdges ].find( candidateEdge => {
        return candidateEdge.startPoint.equalsEpsilon( edge.endPoint, epsilon ) &&
               candidateEdge.endPoint.equalsEpsilon( edge.startPoint, epsilon );
      } );
      if ( opposite ) {
        remainingEdges.delete( opposite );
      }
      else {
        outputEdges.push( edge );
      }
    }

    return outputEdges;
  }

  public static polygonsToShape( polygons: Vector2[][] ): Shape {
    const shape = new Shape();

    polygons.forEach( polygon => {
      shape.moveToPoint( polygon[ 0 ] );
      for ( let i = 1; i < polygon.length; i++ ) {
        shape.lineToPoint( polygon[ i ] );
      }
      shape.close();
    } );

    return shape;
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

    // TODO: return zero for when we would return NaN?

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

  public serialize(): SerializedLinearEdge {
    return {
      startPoint: { x: this.startPoint.x, y: this.startPoint.y },
      endPoint: { x: this.endPoint.x, y: this.endPoint.y },
      containsFakeCorner: this.containsFakeCorner
    };
  }

  public static deserialize( obj: SerializedLinearEdge ): LinearEdge {
    return new LinearEdge(
      new Vector2( obj.startPoint.x, obj.startPoint.y ),
      new Vector2( obj.endPoint.x, obj.endPoint.y ),
      obj.containsFakeCorner
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

  /**
   * Given a line segment, returns the distance from the origin to the closest point on the line segment.
   */
  public static evaluateClosestDistanceToOrigin( p0x: number, p0y: number, p1x: number, p1y: number ): number {
    const dx = p1x - p0x;
    const dy = p1y - p0y;
    const dMagnitude = Math.sqrt( dx * dx + dy * dy );

    // Normalized delta (start => end)
    const normalizedDX = dx / dMagnitude;
    const normalizedDY = dy / dMagnitude;

    // dot-products of our normalized delta with the start points. This is essentially projecting our start/end points
    // onto a line that is parallel to start-end, but GOES THROUGH THE ORIGIN.
    const startU = p0x * normalizedDX + p0y * normalizedDY;
    const endU = p1x * normalizedDX + p1y * normalizedDY;

    // If the signs are different, then the projection of our segment goes THROUGH the origin, which means that the
    // closest point is in the middle of the line segment
    if ( startU * endU < 0 ) {
      // Normalized perpendicular to start => end
      const perpendicularX = -normalizedDY;
      const perpendicularY = normalizedDX;

      // We can essentially look now at things from the perpendicular orientation. If we take a perpendicular vector
      // that is normalized, its dot-product with both the start and ending point will be the distance to the line
      // (it should be the same).
      return Math.abs( p0x * perpendicularX + p0y * perpendicularY );
    }
    else {
      // Endpoint is the closest, just compute the distance to the closer point (which we can identify by its projection
      // being closer to the origin).
      return Math.abs( startU ) < Math.abs( endU )
             ? Math.sqrt( p0x * p0x + p0y * p0y )
             : Math.sqrt( p1x * p1x + p1y * p1y );
    }
  }
}

scenery.register( 'LinearEdge', LinearEdge );

export type SerializedLinearEdge = {
  startPoint: { x: number; y: number };
  endPoint: { x: number; y: number };
  containsFakeCorner: boolean;
};
