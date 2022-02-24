// Copyright 2013-2021, University of Colorado Boulder

/**
 * Tracks a stylus ('pen') or something with tilt and pressure information
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import { scenery, Pointer } from '../imports.js';

class Pen extends Pointer {

  // For tracking which pen is which
  id: number;

  constructor( id: number, point: Vector2, event: Event ) {
    super( point, true, 'pen' ); // true: pen pointers always start in the down state

    this.id = id;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Created ${this.toString()}` );
  }

  /**
   * Sets information in this Pen for a given move. (scenery-internal)
   *
   * @returns Whether the point changed
   */
  move( point: Vector2, event: Event ): boolean {
    const pointChanged = this.hasPointChanged( point );

    this.point = point;
    return pointChanged;
  }

  /**
   * Sets information in this Pen for a given end. (scenery-internal)
   *
   * @returns Whether the point changed
   */
  end( point: Vector2, event: Event ): boolean {
    const pointChanged = this.hasPointChanged( point );

    this.point = point;
    this.isDown = false;
    return pointChanged;
  }

  /**
   * Sets information in this Pen for a given cancel. (scenery-internal)
   *
   * @returns Whether the point changed
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
    return `Pen#${this.id}`;
  }

  isTouchLike(): boolean {
    return true;
  }
}

scenery.register( 'Pen', Pen );

export default Pen;