// Copyright 2023, University of Colorado Boulder

/**
 * Handles finding intersections between IntegerEdges (will push RationalIntersections into the edge's intersections
 * arrays)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigIntVector2, BigRational, IntegerEdge, IntersectionPoint, RationalIntersection, scenery } from '../../../imports.js';

export default class LineIntersector {
  private static processIntegerEdgeIntersection( edgeA: IntegerEdge, edgeB: IntegerEdge ): void {
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
  }

  public static edgeIntersectionQuadratic( integerEdges: IntegerEdge[] ): void {
    // Compute intersections
    // TODO: improve on the quadratic!!!!
    // similar to BoundsIntersectionFilter.quadraticIntersect( integerBounds, integerEdges, ( edgeA, edgeB ) => {
    for ( let i = 0; i < integerEdges.length; i++ ) {
      const edgeA = integerEdges[ i ];
      const boundsA = edgeA.bounds;
      const xAEqual = edgeA.x0 === edgeA.x1;
      const yAEqual = edgeA.y0 === edgeA.y1;

      for ( let j = i + 1; j < integerEdges.length; j++ ) {
        const edgeB = integerEdges[ j ];
        const boundsB = edgeB.bounds;
        const someXEqual = xAEqual || edgeB.x0 === edgeB.x1;
        const someYEqual = yAEqual || edgeB.y0 === edgeB.y1;

        // Bounds min/max for overlap checks
        const minX = Math.max( boundsA.minX, boundsB.minX );
        const minY = Math.max( boundsA.minY, boundsB.minY );
        const maxX = Math.min( boundsA.maxX, boundsB.maxX );
        const maxY = Math.min( boundsA.maxY, boundsB.maxY );

        // If one of the segments is (e.g.) vertical, we'll need to allow checks for overlap ONLY on the x value, otherwise
        // we can have a strict inequality check. This also applies to horizontal segments and the y value.
        // The reason this is OK is because if the segments are both (e.g.) non-vertical, then if the bounds only meet
        // at a single x value (and not a continuos area of overlap), THEN the only intersection would be at the
        // endpoints (which we would filter out and not want anyway).
        if (
          someXEqual ? ( maxX >= minX ) : ( maxX > minX ) &&
          someYEqual ? ( maxY >= minY ) : ( maxY > minY )
        ) {
          LineIntersector.processIntegerEdgeIntersection( edgeA, edgeB );
        }
      }
    }
  }
}

scenery.register( 'LineIntersector', LineIntersector );
