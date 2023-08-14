// Copyright 2023, University of Colorado Boulder

/**
 * A line segment (between two points).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../imports.js';
import Vector2 from '../../../../dot/js/Vector2.js';

export default class LinearEdge {
  public constructor( public readonly startPoint: Vector2, public readonly endPoint: Vector2 ) {
    assert && assert( !startPoint.equals( endPoint ) );
  }

  /**
   * If you take the sum of these for a closed polygon, it should be the area of the polygon.
   */
  public getLineIntegralArea(): number {
    return 0.5 * ( this.endPoint.x + this.startPoint.x ) * ( this.endPoint.y - this.startPoint.y );
  }

  /**
   * If you take the sum of these for a closed polygon and DIVIDE IT by the area, it should be the centroid of the
   * polygon.
   */
  public getLineIntegralPartialCentroid(): Vector2 {
    const base = ( 1 / 6 ) * ( this.startPoint.x * this.endPoint.y - this.endPoint.x * this.startPoint.y );

    return new Vector2(
      ( this.startPoint.x + this.endPoint.x ) * base,
      ( this.startPoint.y + this.endPoint.y ) * base
    );
  }

  // TODO: use this to check all of our LinearEdge computations
  /**
   * If you take the sum of these for a closed polygon, it should be zero (used to check computations).
   */
  public getLineIntegralZero(): number {
    return ( this.startPoint.x - 0.1396 ) * ( this.startPoint.y + 1.422 ) -
           ( this.endPoint.x - 0.1396 ) * ( this.endPoint.y + 1.422 );
  }
}

scenery.register( 'LinearEdge', LinearEdge );
