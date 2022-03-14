// Copyright 2022, University of Colorado Boulder

/**
 * Pooled structure to record batched events efficiently. How it calls the callback is based on the type
 * (pointer/mspointer/touch/mouse). There is one BatchedDOMEvent for each DOM Event (not for each touch).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Enumeration from '../../../phet-core/js/Enumeration.js';
import EnumerationValue from '../../../phet-core/js/EnumerationValue.js';
import Pool, { IPoolable } from '../../../phet-core/js/Pool.js';
import { Input, scenery } from '../imports.js';

export type BatchedDOMEventCallback = ( ...args: any[] ) => void;

export class BatchedDOMEventType extends EnumerationValue {
  static POINTER_TYPE = new BatchedDOMEventType();
  static MS_POINTER_TYPE = new BatchedDOMEventType();
  static TOUCH_TYPE = new BatchedDOMEventType();
  static MOUSE_TYPE = new BatchedDOMEventType();
  static WHEEL_TYPE = new BatchedDOMEventType();

  static enumeration = new Enumeration( BatchedDOMEventType, {
    phetioDocumentation: 'The type of batched event'
  } );
}

export default class BatchedDOMEvent implements IPoolable {

  domEvent!: Event | null;
  type!: BatchedDOMEventType | null;
  callback!: BatchedDOMEventCallback | null;

  constructor( domEvent: Event, type: BatchedDOMEventType, callback: BatchedDOMEventCallback ) {
    this.initialize( domEvent, type, callback );
  }

  initialize( domEvent: Event, type: BatchedDOMEventType, callback: BatchedDOMEventCallback ) {
    // called multiple times due to pooling, this should be re-entrant
    assert && assert( domEvent, 'for some reason, there is no DOM event?' );

    this.domEvent = domEvent;
    this.type = type;
    this.callback = callback;
  }

  run( input: Input ) {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'Running batched event' );
    sceneryLog && sceneryLog.InputEvent && sceneryLog.push();

    const domEvent = this.domEvent!;
    const callback = this.callback!;

    // process whether anything under the pointers changed before running additional input events
    input.validatePointers();

    //OHTWO TODO: switch?
    if ( this.type === BatchedDOMEventType.POINTER_TYPE ) {
      const pointerEvent = domEvent as PointerEvent;
      callback.call( input, pointerEvent.pointerId, pointerEvent.pointerType, input.pointFromEvent( pointerEvent ), pointerEvent );
    }
    else if ( this.type === BatchedDOMEventType.MS_POINTER_TYPE ) {
      const pointerEvent = domEvent as PointerEvent;
      callback.call( input, pointerEvent.pointerId, Input.msPointerType( pointerEvent ), input.pointFromEvent( pointerEvent ), pointerEvent );
    }
    else if ( this.type === BatchedDOMEventType.TOUCH_TYPE ) {
      const touchEvent = domEvent as TouchEvent;
      for ( let i = 0; i < touchEvent.changedTouches.length; i++ ) {
        // according to spec (http://www.w3.org/TR/touch-events/), this is not an Array, but a TouchList
        const touch = touchEvent.changedTouches.item( i )!;

        callback.call( input, touch!.identifier, input.pointFromEvent( touch ), touchEvent );
      }
    }
    else if ( this.type === BatchedDOMEventType.MOUSE_TYPE ) {
      const mouseEvent = domEvent as MouseEvent;
      if ( callback === input.mouseDown ) {
        callback.call( input, null, input.pointFromEvent( mouseEvent ), mouseEvent );
      }
      else {
        callback.call( input, input.pointFromEvent( mouseEvent ), mouseEvent );
      }
    }
    else if ( this.type === BatchedDOMEventType.WHEEL_TYPE ) {
      callback.call( input, domEvent );
    }
    else {
      throw new Error( `bad type value: ${this.type}` );
    }

    sceneryLog && sceneryLog.InputEvent && sceneryLog.pop();
  }

  /**
   * Releases references
   */
  dispose() {
    // clear our references
    this.domEvent = null;
    this.callback = null;
    this.freeToPool();
  }

  freeToPool() {
    BatchedDOMEvent.pool.freeToPool( this );
  }

  static pool = new Pool( BatchedDOMEvent );
}

scenery.register( 'BatchedDOMEvent', BatchedDOMEvent );
