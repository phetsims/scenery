// Copyright 2023, University of Colorado Boulder

/**
 * Simplification of a polygon for clipping output (compacts equal or axis-aligned-collinear points).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';
import Vector2 from '../../../../dot/js/Vector2.js';

export default class ClipSimplifier {

  private points: Vector2[] = [];

  public constructor() {
    // NOTHING NEEDED
  }

  public reset(): void {
    this.points = [];
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

    // TODO: pooling?
    this.points.push( new Vector2( x, y ) );
  }

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
    } while ( changed );

    // Clear out to an empty array if we won't have enough points to have any area
    if ( this.points.length <= 2 ) {
      this.points.length = 0;
    }

    return this.points;
  }
}

scenery.register( 'ClipSimplifier', ClipSimplifier );