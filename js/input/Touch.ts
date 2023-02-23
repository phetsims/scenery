// Copyright 2013-2023, University of Colorado Boulder

/**
 * Tracks a single touch point
 *
 * IE guidelines for Touch-friendly sites: http://blogs.msdn.com/b/ie/archive/2012/04/20/guidelines-for-building-touch-friendly-sites.aspx
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import { Pointer, scenery } from '../imports.js';

export default class Touch extends Pointer {

  // For tracking which touch is which
  public id: number;

  public constructor( id: number, point: Vector2, event: Event ) {
    super( point, 'touch' ); // true: touches always start in the down state

    this.id = id;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Created ${this.toString()}` );
  }

  /**
   * Sets information in this Touch for a given touch move. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  public move( point: Vector2 ): boolean {
    const pointChanged = this.hasPointChanged( point );

    this.point = point;

    return pointChanged;
  }

  /**
   * Returns an improved string representation of this object.
   */
  public override toString(): string {
    return `Touch#${this.id}`;
  }

  public override isTouchLike(): boolean {
    return true;
  }
}

scenery.register( 'Touch', Touch );
