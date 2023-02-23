// Copyright 2013-2023, University of Colorado Boulder

/**
 * Tracks the mouse state
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import Vector3 from '../../../dot/js/Vector3.js';
import { Pointer, scenery } from '../imports.js';

export default class Mouse extends Pointer {

  // Since we need to track the mouse's pointer id occasionally
  public id: number | null;

  // @deprecated, see https://github.com/phetsims/scenery/issues/803
  public leftDown: boolean; // @deprecated
  public middleDown: boolean; // @deprecated
  public rightDown: boolean; // @deprecated

  // Mouse wheel delta for the last event, see https://developer.mozilla.org/en-US/docs/Web/Events/wheel
  public wheelDelta: Vector3;

  // Mouse wheel mode for the last event (0: pixels, 1: lines, 2: pages), see
  // https://developer.mozilla.org/en-US/docs/Web/Events/wheel
  public wheelDeltaMode: number;

  public constructor( point: Vector2 ) {
    super( point, 'mouse' );

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
  public override down( event: Event ): void {
    assert && assert( event instanceof MouseEvent ); // eslint-disable-line no-simple-type-checking-assertions
    const mouseEvent = event as MouseEvent;

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
    return super.down( event );
  }

  /**
   * Sets information in this Mouse for a given mouse up. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  public override up( point: Vector2, event: Event ): boolean {
    assert && assert( event instanceof MouseEvent ); // eslint-disable-line no-simple-type-checking-assertions
    const mouseEvent = event as MouseEvent;

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

    return super.up( point, event );
  }

  /**
   * Sets information in this Mouse for a given mouse move. (scenery-internal)
   *
   * @returns - Whether the point changed
   */
  public move( point: Vector2 ): boolean {
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
  public over( point: Vector2 ): boolean {
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
  public out( point: Vector2 ): boolean {
    const pointChanged = this.hasPointChanged( point );
    point && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( `mouse out at ${point.toString()}` );

    return pointChanged;
  }

  /**
   * Sets information in this Mouse for a given mouse wheel. (scenery-internal)
   */
  public wheel( event: Event ): void {
    assert && assert( event instanceof WheelEvent ); // eslint-disable-line no-simple-type-checking-assertions
    const wheelEvent = event as WheelEvent;

    this.wheelDelta.setXYZ( wheelEvent.deltaX, wheelEvent.deltaY, wheelEvent.deltaZ );
    this.wheelDeltaMode = wheelEvent.deltaMode;
  }

  /**
   * Returns an improved string representation of this object.
   */
  public override toString(): string {
    return 'Mouse'; // there is only one
  }
}

scenery.register( 'Mouse', Mouse );
