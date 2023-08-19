// Copyright 2023, University of Colorado Boulder

/**
 * Code to split integer edges into rational edges
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigRational, BigRationalVector2, IntegerEdge, RationalHalfEdge, scenery } from '../../../imports.js';

export default class LineSplitter {
  public static splitIntegerEdges( integerEdges: IntegerEdge[] ): RationalHalfEdge[] {
    let edgeIdCounter = 0;
    const rationalHalfEdges: RationalHalfEdge[] = [];

    // TODO: reduce closures
    integerEdges.forEach( integerEdge => {
      const points = [
        new BigRationalVector2( BigRational.whole( integerEdge.x0 ), BigRational.whole( integerEdge.y0 ) )
      ];

      let lastT = BigRational.ZERO;

      integerEdge.intersections.sort( ( a, b ) => {
        // TODO: we'll need to map this over with functions
        return a.t.compareCrossMul( b.t );
      } );

      // Deduplicate
      integerEdge.intersections.forEach( intersection => {
        if ( !lastT.equals( intersection.t ) ) {
          points.push( intersection.point );
        }
        lastT = intersection.t;
      } );

      points.push( new BigRationalVector2( BigRational.whole( integerEdge.x1 ), BigRational.whole( integerEdge.y1 ) ) );

      for ( let i = 0; i < points.length - 1; i++ ) {
        const p0 = points[ i ];
        const p1 = points[ i + 1 ];

        // We will remove degenerate edges now, so during the deduplication we won't collapse them together
        if ( !p0.equals( p1 ) ) {
          const edgeId = edgeIdCounter++;
          const forwardEdge = new RationalHalfEdge( edgeId, p0, p1 );
          const reverseEdge = new RationalHalfEdge( edgeId, p1, p0 );
          forwardEdge.reversed = reverseEdge;
          reverseEdge.reversed = forwardEdge;
          forwardEdge.windingMap.addWindingNumber( integerEdge.renderPath, 1 );
          reverseEdge.windingMap.addWindingNumber( integerEdge.renderPath, -1 );

          const deltaX = integerEdge.x1 - integerEdge.x0;
          const deltaY = integerEdge.y1 - integerEdge.y0;

          // We compute slope here due to rational precision (while it would be possible to create a larger rational
          // number later and reduce it, here we're starting with integers, so we don't have to do as much).
          const discriminator = deltaY === 0 ? ( deltaX > 0 ? 0 : 2 ) : ( deltaY > 0 ? 1 : 3 );
          const slope = deltaY === 0 ? BigRational.ZERO : new BigRational( deltaX, deltaY ).reduced();

          // We store the slope and discriminator here, as that allows us to tell the order-difference between two
          // edges that have one point the same. This works here, because we have already broken lines up at the
          // endpoints in the case of overlap, so that if it has the same start point, discriminator and slope, then it
          // WILL have the same end point, and thus will be the same effective edge.
          forwardEdge.discriminator = discriminator;
          reverseEdge.discriminator = ( discriminator + 2 ) % 4;
          forwardEdge.slope = slope;
          reverseEdge.slope = slope;

          rationalHalfEdges.push( forwardEdge );
          rationalHalfEdges.push( reverseEdge );
        }
      }
    } );
    return rationalHalfEdges;
  }
}

scenery.register( 'LineSplitter', LineSplitter );
