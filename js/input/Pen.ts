// Copyright 2013-2023, University of Colorado Boulder

/**
 * Tracks a stylus ('pen') or something with tilt and pressure information
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import { Pointer, scenery } from '../imports.js';

export default class Pen extends Pointer {

  // For tracking which pen is which
  public id: number;

  public constructor( id: number, point: Vector2, event: Event ) {
    super( point, 'pen' ); // true: pen pointers always start in the down state

    this.id = id;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Created ${this.toString()}` );
  }

  /**
   * Sets information in this Pen for a given move. (scenery-internal)
   *
   * @returns Whether the point changed
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
    return `Pen#${this.id}`;
  }

  public override isTouchLike(): boolean {
    return true;
  }
}

scenery.register( 'Pen', Pen );
