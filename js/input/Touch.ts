// Copyright 2013-2022, University of Colorado Boulder

/**
 * Tracks a single touch point
 *
 * IE guidelines for Touch-friendly sites: http://blogs.msdn.com/b/ie/archive/2012/04/20/guidelines-for-building-touch-friendly-sites.aspx
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import { scenery, Pointer } from '../imports.js';

class Touch extends Pointer {

  // For tracking which touch is which
  id: number;

  constructor( id: number, point: Vector2, event: Event ) {
    super( point, true, 'touch' ); // true: touches always start in the down state

    this.id = id;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Created ${this.toString()}` );
  }


  /**
   * Sets information in this Touch for a given touch move. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  move( point: Vector2, event: Event ): boolean {
    const pointChanged = this.hasPointChanged( point );

    this.point = point;
    return pointChanged;
  }

  /**
   * Sets information in this Touch for a given touch end. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  end( point: Vector2, event: Event ): boolean {
    const pointChanged = this.hasPointChanged( point );

    this.point = point;
    this.isDown = false;
    return pointChanged;
  }

  /**
   * Sets information in this Touch for a given touch cancel. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  cancel( point: Vector2, event: Event ): boolean {
    const pointChanged = this.hasPointChanged( point );

    this.point = point;
    this.isDown = false;
    return pointChanged;
  }

  /**
   * Returns an improved string representation of this object.
   */
  toString(): string {
    return `Touch#${this.id}`;
  }

  isTouchLike(): boolean {
    return true;
  }
}

scenery.register( 'Touch', Touch );
export default Touch;