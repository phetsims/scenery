// Copyright 2023, University of Colorado Boulder

/**
 * A line-segment edge with integer coordinates, as part of the rendering
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { BoundedSubpath, PolygonClipping, RationalIntersection, RenderPath, scenery } from '../../../imports.js';
import Bounds2 from '../../../../../dot/js/Bounds2.js';
import Utils from '../../../../../dot/js/Utils.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

export default class IntegerEdge {

  public readonly bounds: Bounds2;
  public readonly intersections: RationalIntersection[] = [];

  public constructor(
    public readonly renderPath: RenderPath,
    public readonly x0: number,
    public readonly y0: number,
    public readonly x1: number,
    public readonly y1: number
  ) {
    assert && assert( Number.isInteger( x0 ) );
    assert && assert( Number.isInteger( y0 ) );
    assert && assert( Number.isInteger( x1 ) );
    assert && assert( Number.isInteger( y1 ) );
    assert && assert( x0 !== x1 || y0 !== y1 );

    // TODO: maybe don't compute this here? Can we compute it in the other work?
    this.bounds = new Bounds2(
      Math.min( x0, x1 ),
      Math.min( y0, y1 ),
      Math.max( x0, x1 ),
      Math.max( y0, y1 )
    );
  }

  public hasBoundsIntersectionWith( other: IntegerEdge ): boolean {
    return IntegerEdge.hasBoundsIntersection(
      this.bounds,
      other.bounds,
      this.x0 === this.x1 || other.x0 === other.x1,
      this.y0 === this.y1 || other.y0 === other.y1
    );
  }

  // If one of the segments is (e.g.) vertical, we'll need to allow checks for overlap ONLY on the x value, otherwise
  // we can have a strict inequality check. This also applies to horizontal segments and the y value.
  // The reason this is OK is because if the segments are both (e.g.) non-vertical, then if the bounds only meet
  // at a single x value (and not a continuos area of overlap), THEN the only intersection would be at the
  // endpoints (which we would filter out and not want anyway).
  public static hasBoundsIntersection( boundsA: Bounds2, boundsB: Bounds2, someXEqual: boolean, someYEqual: boolean ): boolean {
    // Bounds min/max for overlap checks
    const minX = Math.max( boundsA.minX, boundsB.minX );
    const minY = Math.max( boundsA.minY, boundsB.minY );
    const maxX = Math.min( boundsA.maxX, boundsB.maxX );
    const maxY = Math.min( boundsA.maxY, boundsB.maxY );

    return ( someXEqual ? ( maxX >= minX ) : ( maxX > minX ) ) && ( someYEqual ? ( maxY >= minY ) : ( maxY > minY ) );
  }

  public static createTransformed( path: RenderPath, toIntegerMatrix: Matrix3, p0: Vector2, p1: Vector2 ): IntegerEdge | null {
    const m00 = toIntegerMatrix.m00();
    const m01 = toIntegerMatrix.m01();
    const m02 = toIntegerMatrix.m02();
    const m10 = toIntegerMatrix.m10();
    const m11 = toIntegerMatrix.m11();
    const m12 = toIntegerMatrix.m12();
    const x0 = Utils.roundSymmetric( p0.x * m00 + p0.y * m01 + m02 );
    const y0 = Utils.roundSymmetric( p0.x * m10 + p0.y * m11 + m12 );
    const x1 = Utils.roundSymmetric( p1.x * m00 + p1.y * m01 + m02 );
    const y1 = Utils.roundSymmetric( p1.x * m10 + p1.y * m11 + m12 );
    if ( x0 !== x1 || y0 !== y1 ) {
      return new IntegerEdge( path, x0, y0, x1, y1 );
    }
    else {
      return null;
    }
  }

  /**
   * Returns a list of integer edges (tagged with their respective RenderPaths) that are clipped to within the given
   * bounds.
   *
   * Since we also need to apply the to-integer-coordinate-frame conversion at the same time, this step is included.
   */
  public static clipScaleToIntegerEdges( boundedSubpaths: BoundedSubpath[], bounds: Bounds2, toIntegerMatrix: Matrix3 ): IntegerEdge[] {
    const integerEdges: IntegerEdge[] = [];

    for ( let i = 0; i < boundedSubpaths.length; i++ ) {
      const boundedSubpath = boundedSubpaths[ i ];
      const subpath = boundedSubpath.subpath;

      if ( !bounds.intersectsBounds( boundedSubpath.bounds ) ) {
        continue;
      }

      const goesOutsideBounds = !bounds.containsBounds( boundedSubpath.bounds );

      // NOTE: This is a variant that will fully optimize out "doesn't contribute anything" bits to an empty array
      // If a path is fully outside of the clip region, we won't create integer edges out of it.
      // TODO: Optimize our allocations or other parts so that we don't always create a ton of new vectors here
      const clippedSubpath = goesOutsideBounds ? PolygonClipping.boundsClipPolygon( subpath, bounds ) : subpath;

      for ( let k = 0; k < clippedSubpath.length; k++ ) {
        // TODO: when micro-optimizing, improve this pattern so we only have one access each iteration
        const p0 = clippedSubpath[ k ];
        const p1 = clippedSubpath[ ( k + 1 ) % clippedSubpath.length ];
        const edge = IntegerEdge.createTransformed( boundedSubpath.path, toIntegerMatrix, p0, p1 );
        if ( edge !== null ) {
          integerEdges.push( edge );
        }
      }
    }

    return integerEdges;
  }

  /**
   * Returns a list of integer edges (tagged with their respective RenderPaths) that are transformed to within the
   * integer coordinates.
   */
  public static scaleToIntegerEdges( paths: RenderPath[], toIntegerMatrix: Matrix3 ): IntegerEdge[] {
    const integerEdges: IntegerEdge[] = [];
    for ( let i = 0; i < paths.length; i++ ) {
      const path = paths[ i ];

      for ( let j = 0; j < path.subpaths.length; j++ ) {
        const subpath = path.subpaths[ j ];

        for ( let k = 0; k < subpath.length; k++ ) {
          // TODO: when micro-optimizing, improve this pattern so we only have one access each iteration
          const p0 = subpath[ k ];
          const p1 = subpath[ ( k + 1 ) % subpath.length ];
          const edge = IntegerEdge.createTransformed( path, toIntegerMatrix, p0, p1 );
          if ( edge !== null ) {
            integerEdges.push( edge );
          }
        }
      }
    }
    return integerEdges;
  }
}

scenery.register( 'IntegerEdge', IntegerEdge );
