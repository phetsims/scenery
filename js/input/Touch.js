// Copyright 2013-2020, University of Colorado Boulder

/**
 * Tracks a single touch point
 *
 * IE guidelines for Touch-friendly sites: http://blogs.msdn.com/b/ie/archive/2012/04/20/guidelines-for-building-touch-friendly-sites.aspx
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import Pointer from './Pointer.js'; // extends Pointer

class Touch extends Pointer {
  /**
   * @param {number} id
   * @param {Vector2} point
   * @param {Event} event
   */
  constructor( id, point, event ) {
    super( point, true, 'touch' ); // true: touches always start in the down state

    // @public {number} - For tracking which touch is which
    this.id = id;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Created ' + this.toString() );
  }


  /**
   * Sets information in this Touch for a given touch move.
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
   * Sets information in this Touch for a given touch end.
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
   * Sets information in this Touch for a given touch cancel.
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
    return 'Touch#' + this.id;
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

scenery.register( 'Touch', Touch );
export default Touch;