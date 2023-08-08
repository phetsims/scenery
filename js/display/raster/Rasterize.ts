// Copyright 2023, University of Colorado Boulder

/**
 * Test rasterization
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BigIntVector2, BigRational, BigRationalVector2, BoundsIntersectionFilter, IntersectionPoint, PolygonClipping, RenderPathProgram, RenderProgram, scenery } from '../../imports.js';
import { RenderPath } from './RenderProgram.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import Utils from '../../../../dot/js/Utils.js';

class RationalIntersection {
  public constructor( public readonly t: BigRational, public readonly point: BigRationalVector2 ) {}
}

class IntegerEdge {

  public readonly bounds: Bounds2;
  public readonly intersections: RationalIntersection[] = [];

  public constructor(
    public readonly renderPath: RenderPath,
    public readonly x0: number,
    public readonly y0: number,
    public readonly x1: number,
    public readonly y1: number
  ) {
    // TODO: maybe don't compute this here? Can we compute it in the other work?
    this.bounds = new Bounds2(
      Math.min( x0, x1 ),
      Math.min( y0, y1 ),
      Math.max( x0, x1 ),
      Math.max( y0, y1 )
    );
  }
}

class RationalEdge {
  public constructor(
    public readonly renderPath: RenderPath,
    public readonly p0: BigRationalVector2,
    public readonly p1: BigRationalVector2
  ) {}
}

export default class Rasterize {
  public static rasterizeRenderProgram( renderProgram: RenderProgram, bounds: Bounds2 ): void {

    assert && assert( Number.isInteger( bounds.left ) && Number.isInteger( bounds.top ) && Number.isInteger( bounds.right ) && Number.isInteger( bounds.bottom ) );

    // const imageData = new ImageData( bounds.width, bounds.height, { colorSpace: 'srgb' } );

    const scale = Math.pow( 2, 20 - Math.ceil( Math.log2( Math.max( bounds.width, bounds.height ) ) ) );

    const paths: RenderPath[] = [];
    renderProgram.depthFirst( program => {
      if ( program instanceof RenderPathProgram && program.path !== null ) {
        paths.push( program.path );
      }
    } );

    const integerBounds = new Bounds2(
      Utils.roundSymmetric( bounds.minX * scale ),
      Utils.roundSymmetric( bounds.minY * scale ),
      Utils.roundSymmetric( bounds.maxX * scale ),
      Utils.roundSymmetric( bounds.maxY * scale )
    );

    const integerEdges: IntegerEdge[] = [];

    paths.forEach( path => {
      path.subpaths.forEach( subpath => {
        const clippedSubpath = PolygonClipping.boundsClipPolygon( subpath, bounds );

        for ( let i = 0; i < clippedSubpath.length; i++ ) {
          const p0 = clippedSubpath[ i ];
          const p1 = clippedSubpath[ ( i + 1 ) % clippedSubpath.length ];
          const x0 = Utils.roundSymmetric( p0.x * scale );
          const y0 = Utils.roundSymmetric( p0.y * scale );
          const x1 = Utils.roundSymmetric( p1.x * scale );
          const y1 = Utils.roundSymmetric( p1.y * scale );
          integerEdges.push( new IntegerEdge( path, x0, y0, x1, y1 ) );
        }
      } );
    } );

    // Compute intersections
    BoundsIntersectionFilter.quadraticIntersect( integerBounds, integerEdges, ( edgeA, edgeB ) => {
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

        if ( !t0.equals( BigRational.ZERO ) && !t0.equals( BigRational.ONE ) ) {
          edgeA.intersections.push( new RationalIntersection( t0, point ) );
        }
        if ( !t1.equals( BigRational.ZERO ) && !t1.equals( BigRational.ONE ) ) {
          edgeB.intersections.push( new RationalIntersection( t1, point ) );
        }
      }
    } );

    const rationalEdges: RationalEdge[] = [];
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

      points.push( ...integerEdge.intersections.map( intersection => intersection.point ) );

      points.push( new BigRationalVector2( BigRational.whole( integerEdge.x1 ), BigRational.whole( integerEdge.y1 ) ) );

      for ( let i = 0; i < points.length; i++ ) {
        const p0 = points[ i ];
        const p1 = points[ ( i + 1 ) % points.length ];

        rationalEdges.push( new RationalEdge( integerEdge.renderPath, p0, p1 ) );
      }
    } );
  }
}

scenery.register( 'Rasterize', Rasterize );
