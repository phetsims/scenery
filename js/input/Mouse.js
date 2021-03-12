// Copyright 2013-2020, University of Colorado Boulder

/**
 * Tracks the mouse state
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector3 from '../../../dot/js/Vector3.js';
import scenery from '../scenery.js';
import Pointer from './Pointer.js';

class Mouse extends Pointer {
  constructor() {
    super( null, false, 'mouse' );

    // @public {number|null} - Since we need to track the mouse's pointer id occasionally
    this.id = null;

    // @public {boolean} @deprecated, see https://github.com/phetsims/scenery/issues/803
    this.leftDown = false;
    this.middleDown = false;
    this.rightDown = false;

    // @public {Vector3} - Mouse wheel delta for the last event, see
    // https://developer.mozilla.org/en-US/docs/Web/Events/wheel
    this.wheelDelta = new Vector3( 0, 0, 0 );

    // @public {number} - Mouse wheel mode for the last event (0: pixels, 1: lines, 2: pages), see
    // https://developer.mozilla.org/en-US/docs/Web/Events/wheel
    this.wheelDeltaMode = 0;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Created ${this.toString()}` );
  }

  /**
   * Sets information in this Mouse for a given mouse down.
   * @public (scenery-internal)
   *
   * @param {Vector2} point
   * @param {Event} event
   * @returns {boolean} - Whether the point changed
   */
  down( point, event ) {
    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse down at ${point.toString()}` );

    this.point = point;
    this.isDown = true;
    switch( event.button ) {
      case 0:
        this.leftDown = true;
        break;
      case 1:
        this.middleDown = true;
        break;
      case 2:
        this.rightDown = true;
        break;
      default:
      // no-op until we refactor things, see https://github.com/phetsims/scenery/issues/813
    }
    return pointChanged;
  }

  /**
   * Sets information in this Mouse for a given mouse up.
   * @public (scenery-internal)
   *
   * @param {Vector2} point
   * @param {Event} event
   * @returns {boolean} - Whether the point changed
   */
  up( point, event ) {
    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse up at ${point.toString()}` );

    this.point = point;
    this.isDown = false;
    switch( event.button ) {
      case 0:
        this.leftDown = false;
        break;
      case 1:
        this.middleDown = false;
        break;
      case 2:
        this.rightDown = false;
        break;
      default:
      // no-op until we refactor things, see https://github.com/phetsims/scenery/issues/813
    }
    return pointChanged;
  }

  /**
   * Sets information in this Mouse for a given mouse move.
   * @public (scenery-internal)
   *
   * @param {Vector2} point
   * @param {Event} event
   * @returns {boolean} - Whether the point changed
   */
  move( point, event ) {
    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse move at ${point.toString()}` );

    this.point = point;
    return pointChanged;
  }

  /**
   * Sets information in this Mouse for a given mouse over.
   * @public (scenery-internal)
   *
   * @param {Vector2} point
   * @param {Event} event
   * @returns {boolean} - Whether the point changed
   */
  over( point, event ) {
    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse over at ${point.toString()}` );

    this.point = point;
    return pointChanged;
  }

  /**
   * Sets information in this Mouse for a given mouse out.
   * @public (scenery-internal)
   *
   * @param {Vector2} point
   * @param {Event} event
   * @returns {boolean} - Whether the point changed
   */
  out( point, event ) {
    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse out at ${point.toString()}` );

    // TODO: how to handle the mouse out-of-bounds
    this.point = null;
    return pointChanged;
  }


  /**
   * Sets information in this Mouse for a given mouse wheel.
   * @public (scenery-internal)
   *
   * @param {Event} event
   */
  wheel( event ) {
    this.wheelDelta.setXYZ( event.deltaX, event.deltaY, event.deltaZ );
    this.wheelDeltaMode = event.deltaMode;
  }

  /**
   * Returns an improved string representation of this object.
   * @public
   * @override
   *
   * @returns {string}
   */
  toString() {
    return 'Mouse'; // there is only one
  }
}

scenery.register( 'Mouse', Mouse );
export default Mouse;