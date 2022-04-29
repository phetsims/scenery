// Copyright 2013-2022, University of Colorado Boulder

/**
 * Tracks the mouse state
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import Vector3 from '../../../dot/js/Vector3.js';
import { scenery, Pointer } from '../imports.js';

export default class Mouse extends Pointer {

  // Since we need to track the mouse's pointer id occasionally
  id: number | null;

  // @deprecated, see https://github.com/phetsims/scenery/issues/803
  leftDown: boolean; // @deprecated
  middleDown: boolean; // @deprecated
  rightDown: boolean; // @deprecated

  // Mouse wheel delta for the last event, see https://developer.mozilla.org/en-US/docs/Web/Events/wheel
  wheelDelta: Vector3;

  // Mouse wheel mode for the last event (0: pixels, 1: lines, 2: pages), see
  // https://developer.mozilla.org/en-US/docs/Web/Events/wheel
  wheelDeltaMode: number;

  constructor( point: Vector2 ) {
    super( point, false, 'mouse' );

    this.id = null;
    this.leftDown = false;
    this.middleDown = false;
    this.rightDown = false;
    this.wheelDelta = new Vector3( 0, 0, 0 );
    this.wheelDeltaMode = 0;

    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( `Created ${this.toString()}` );
  }

  /**
   * Sets information in this Mouse for a given mouse down. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  down( point: Vector2, event: Event ): boolean {
    assert && assert( event instanceof MouseEvent );
    const mouseEvent = event as MouseEvent;

    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse down at ${point.toString()}` );

    this.point = point;
    this.isDown = true;
    switch( mouseEvent.button ) {
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
   * Sets information in this Mouse for a given mouse up. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  up( point: Vector2, event: Event ): boolean {
    assert && assert( event instanceof MouseEvent );
    const mouseEvent = event as MouseEvent;

    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse up at ${point.toString()}` );

    this.point = point;
    this.isDown = false;
    switch( mouseEvent.button ) {
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
   * Sets information in this Mouse for a given mouse move. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  move( point: Vector2, event: Event ): boolean {
    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse move at ${point.toString()}` );

    this.point = point;
    return pointChanged;
  }

  /**
   * Sets information in this Mouse for a given mouse over. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  over( point: Vector2, event: Event ): boolean {
    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse over at ${point.toString()}` );

    this.point = point;
    return pointChanged;
  }

  /**
   * Sets information in this Mouse for a given mouse out. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  out( point: Vector2, event: Event ): boolean {
    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse out at ${point.toString()}` );

    return pointChanged;
  }

  /**
   * Sets information in this Mouse for a given mouse wheel. (scenery-internal)
   */
  wheel( event: Event ): void {
    assert && assert( event instanceof WheelEvent );
    const wheelEvent = event as WheelEvent;

    this.wheelDelta.setXYZ( wheelEvent.deltaX, wheelEvent.deltaY, wheelEvent.deltaZ );
    this.wheelDeltaMode = wheelEvent.deltaMode;
  }

  /**
   * Returns an improved string representation of this object.
   */
  override toString(): string {
    return 'Mouse'; // there is only one
  }
}

scenery.register( 'Mouse', Mouse );
