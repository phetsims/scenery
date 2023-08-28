// Copyright 2023, University of Colorado Boulder

/**
 * Simplification of a polygon for clipping output (compacts equal or axis-aligned-collinear points).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';
import Vector2 from '../../../../../dot/js/Vector2.js';
import Utils from '../../../../../dot/js/Utils.js';
import Matrix3 from '../../../../../dot/js/Matrix3.js';

const collinearEpsilon = 1e-9;

export default class ClipSimplifier {

  private points: Vector2[] = [];

  // NOTE: somewhat less performance when general collinearity is checked
  public constructor( private readonly checkGeneralCollinearity = false ) {
    // NOTHING NEEDED
  }

  public addTransformed( matrix: Matrix3, x: number, y: number ): void {
    this.add(
      matrix.m00() * x + matrix.m01() * y + matrix.m02(),
      matrix.m10() * x + matrix.m11() * y + matrix.m12()
    );
  }

  public addPoint( p: Vector2 ): void {
    this.add( p.x, p.y );
  }

  public add( x: number, y: number ): void {
    if ( this.points.length >= 1 ) {

      const lastPoint = this.points[ this.points.length - 1 ];
      const xEquals = lastPoint.x === x;
      const yEquals = lastPoint.y === y;

      // If we are equal to the last point, NO-OP
      if ( xEquals && yEquals ) {
        return;
      }

      if ( this.points.length >= 2 ) {
        const secondLastPoint = this.points[ this.points.length - 2 ];

        if ( this.checkGeneralCollinearity ) {
          if ( Utils.arePointsCollinear( new Vector2( x, y ), lastPoint, secondLastPoint, collinearEpsilon ) ) {
            lastPoint.x = x;
            lastPoint.y = y;
            return;
          }
        }
        else {
          const secondXEquals = secondLastPoint.x === x;
          const secondYEquals = secondLastPoint.y === y;

          // If we are equal to the second-to-last point, we can just undo our last point
          if ( secondXEquals && secondYEquals ) {
            this.points.pop(); // TODO: pooling freeToPool?
            return;
          }

          // X-collinearity check (if we would have 3 points with the same X, we can just remove the middle one)
          if ( xEquals && secondXEquals ) {
            // Instead of adding new one and removing the middle one, we can just update the last one
            lastPoint.y = y;
            return;
          }

          // Y-collinearity check (if we would have 3 points with the same Y, we can just remove the middle one)
          if ( yEquals && secondYEquals ) {
            // Instead of adding new one and removing the middle one, we can just update the last one
            lastPoint.x = x;
            return;
          }
        }
      }
    }

    // TODO: pooling?
    this.points.push( new Vector2( x, y ) );
  }

  /**
   * Treats the entire list of points so far as a closed polygon, completes simplification (between the start/end),
   * returns the simplified points, AND resets the state for the next time
   */
  public finalize(): Vector2[] {
    // TODO: add more comprehensive testing for this! Tested a simple example

    // TODO: is this complexity worth porting to WGSL?
    // We'll handle our equality and collinearity checks. Because we could have a situation where the end of our points
    // retraces the start of our points (but backwards, is possible since the middle could be fine), we'll actually need
    // iteration to rewind this. Repeatedly check equality/collinearity until we don't change anything.
    let changed: boolean;
    do {
      changed = false;
      // Equality check (start/end)
      if ( this.points.length >= 2 ) {
        const firstPoint = this.points[ 0 ];
        const lastPoint = this.points[ this.points.length - 1 ];

        // If the first and last points are equal, remove the last point
        if ( firstPoint.equals( lastPoint ) ) {
          this.points.pop(); // TODO: pooling freeToPool?
          changed = true;
        }
      }

      // Collinearity check (the first two points, and last two points)
      if ( this.points.length >= 3 ) {
        if ( this.checkGeneralCollinearity ) {
          const firstPoint = this.points[ 0 ];
          const secondPoint = this.points[ 1 ];
          const lastPoint = this.points[ this.points.length - 1 ];
          const secondLastPoint = this.points[ this.points.length - 2 ];

          if ( Utils.arePointsCollinear( secondPoint, firstPoint, lastPoint, collinearEpsilon ) ) {
            this.points.shift();
            changed = true;
          }
          if ( Utils.arePointsCollinear( firstPoint, lastPoint, secondLastPoint, collinearEpsilon ) ) {
            this.points.pop();
            changed = true;
          }
        }
        else {
          // NOTE: It is technically possible that this happens with exactly three points left (that are collinear).
          // This should still work to reduce it, but will "garble" the order. We don't care, since the resulting
          // polygon would have no area.
          const firstPoint = this.points[ 0 ];
          const lastPoint = this.points[ this.points.length - 1 ];

          const xEquals = firstPoint.x === lastPoint.x;
          const yEquals = firstPoint.y === lastPoint.y;

          if ( xEquals || yEquals ) {
            const secondPoint = this.points[ 1 ];
            const secondLastPoint = this.points[ this.points.length - 2 ];

            if (
              ( xEquals && firstPoint.x === secondPoint.x ) ||
              ( yEquals && firstPoint.y === secondPoint.y )
            ) {
              // TODO: We can record the "starting" index, and avoid repeated shifts (that are probably horrible for perf)
              // TODO: See if this is significant, or needed for WGSL
              this.points.shift(); // TODO: pooling freeToPool?
              changed = true;
            }

            if (
              ( xEquals && lastPoint.x === secondLastPoint.x ) ||
              ( yEquals && lastPoint.y === secondLastPoint.y )
            ) {
              this.points.pop(); // TODO: pooling freeToPool?
              changed = true;
            }
          }
        }
      }
    } while ( changed );

    // Clear out to an empty array if we won't have enough points to have any area
    if ( this.points.length <= 2 ) {
      this.points.length = 0;
    }

    // Reset our own list of points, so the next time we add points we don't have to clear out the old ones
    const points = this.points;
    this.points = [];
    return points;
  }

  public static simplifyPolygon( polygon: Vector2[], checkGeneralCollinearity = false ): Vector2[] {
    const simplifier = new ClipSimplifier( checkGeneralCollinearity );
    for ( let i = 0; i < polygon.length; i++ ) {
      simplifier.addPoint( polygon[ i ] );
    }
    return simplifier.finalize();
  }
}

scenery.register( 'ClipSimplifier', ClipSimplifier );
