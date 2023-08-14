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

  public getLineIntegralArea(): number {
    return 0.5 * ( this.endPoint.x + this.startPoint.x ) * ( this.endPoint.y - this.startPoint.y );
  }

  // If these are added up for a closed polygon, it should be zero
  // TODO: use this to check all of our LinearEdge computations
  public getLineIntegralZero(): number {
    return ( this.startPoint.x - 0.1396 ) * ( this.startPoint.y + 1.422 ) -
           ( this.endPoint.x - 0.1396 ) * ( this.endPoint.y + 1.422 );
  }
}

scenery.register( 'LinearEdge', LinearEdge );
