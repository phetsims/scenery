// Copyright 2013-2021, University of Colorado Boulder

/**
 * Tracks a stylus ('pen') or something with tilt and pressure information
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Pointer } from '../imports.js';

class Pen extends Pointer {
  /**
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   */
  constructor( id, point, event ) {
    super( point, true, 'pen' ); // true: pen pointers always start in the down state

    // @public {number} - For tracking which pen is which
    this.id = id;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Created ${this.toString()}` );
  }

  /**
   * Sets information in this Pen for a given move.
   * @public (scenery-internal)
   *
   * @param {Vector2} point
   * @param {Event} event
   * @returns {boolean} - Whether the point changed
   */
  move( point, event ) {
    const pointChanged = this.hasPointChanged( point );

    this.point = point;
    return pointChanged;
  }

  /**
   * Sets information in this Pen for a given end.
   * @public (scenery-internal)
   *
   * @param {Vector2} point
   * @param {Event} event
   * @returns {boolean} - Whether the point changed
   */
  end( point, event ) {
    const pointChanged = this.hasPointChanged( point );

    this.point = point;
    this.isDown = false;
    return pointChanged;
  }

  /**
   * Sets information in this Pen for a given cancel.
   * @public (scenery-internal)
   *
   * @param {Vector2} point
   * @param {Event} event
   * @returns {boolean} - Whether the point changed
   */
  cancel( point, event ) {
    const pointChanged = this.hasPointChanged( point );

    this.point = point;
    this.isDown = false;
    return pointChanged;
  }

  /**
   * Returns an improved string representation of this object.
   * @public
   * @override
   *
   * @returns {string}
   */
  toString() {
    return `Pen#${this.id}`;
  }

  /**
   * @override
   * @public
   *
   * @returns {boolean}
   */
  isTouchLike() {
    return true;
  }
}

scenery.register( 'Pen', Pen );

export default Pen;