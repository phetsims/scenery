// Copyright 2022-2023, University of Colorado Boulder

/**
 * Pooled structure to record batched events efficiently. How it calls the callback is based on the type
 * (pointer/mspointer/touch/mouse). There is one BatchedDOMEvent for each DOM Event (not for each touch).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Enumeration from '../../../phet-core/js/Enumeration.js';
import EnumerationValue from '../../../phet-core/js/EnumerationValue.js';
import Pool, { TPoolable } from '../../../phet-core/js/Pool.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';
import { EventContext, Input, scenery } from '../imports.js';

export type BatchedDOMEventCallback = ( ...args: IntentionalAny[] ) => void;

export class BatchedDOMEventType extends EnumerationValue {
  public static readonly POINTER_TYPE = new BatchedDOMEventType();
  public static readonly MS_POINTER_TYPE = new BatchedDOMEventType();
  public static readonly TOUCH_TYPE = new BatchedDOMEventType();
  public static readonly MOUSE_TYPE = new BatchedDOMEventType();
  public static readonly WHEEL_TYPE = new BatchedDOMEventType();
  public static readonly ALT_TYPE = new BatchedDOMEventType();

  public static readonly enumeration = new Enumeration( BatchedDOMEventType, {
    phetioDocumentation: 'The type of batched event'
  } );
}

export default class BatchedDOMEvent implements TPoolable {

  private eventContext!: EventContext | null;
  private type!: BatchedDOMEventType | null;
  private callback!: BatchedDOMEventCallback | null;

  public constructor( eventContext: EventContext, type: BatchedDOMEventType, callback: BatchedDOMEventCallback ) {
    this.initialize( eventContext, type, callback );
  }

  public initialize( eventContext: EventContext, type: BatchedDOMEventType, callback: BatchedDOMEventCallback ): this {
    // called multiple times due to pooling, this should be re-entrant
    assert && assert( eventContext.domEvent, 'for some reason, there is no DOM event?' );

    this.eventContext = eventContext;
    this.type = type;
    this.callback = callback;

    return this;
  }

  public run( input: Input ): void {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'Running batched event' );
    sceneryLog && sceneryLog.InputEvent && sceneryLog.push();

    const callback = this.callback!;

    // process whether anything under the pointers changed before running additional input events
    input.validatePointers();

    //OHTWO TODO: switch?
    if ( this.type === BatchedDOMEventType.POINTER_TYPE ) {
      const context = this.eventContext as EventContext<PointerEvent>;
      const pointerEvent = context.domEvent;
      callback.call( input, pointerEvent.pointerId, pointerEvent.pointerType, input.pointFromEvent( pointerEvent ), context );
    }
    else if ( this.type === BatchedDOMEventType.MS_POINTER_TYPE ) {
      const context = this.eventContext as EventContext<PointerEvent>;
      const pointerEvent = context.domEvent;
      callback.call( input, pointerEvent.pointerId, Input.msPointerType( pointerEvent ), input.pointFromEvent( pointerEvent ), context );
    }
    else if ( this.type === BatchedDOMEventType.TOUCH_TYPE ) {
      const context = this.eventContext as EventContext<TouchEvent>;
      const touchEvent = context.domEvent;
      for ( let i = 0; i < touchEvent.changedTouches.length; i++ ) {
        // according to spec (http://www.w3.org/TR/touch-events/), this is not an Array, but a TouchList
        const touch = touchEvent.changedTouches.item( i )!;

        callback.call( input, touch.identifier, input.pointFromEvent( touch ), context );
      }
    }
    else if ( this.type === BatchedDOMEventType.MOUSE_TYPE ) {
      const context = this.eventContext as EventContext<MouseEvent>;
      const point = input.pointFromEvent( context.domEvent );
      if ( callback === input.mouseDown ) {
        callback.call( input, null, point, context );
      }
      else {
        callback.call( input, point, context );
      }
    }
    else if ( this.type === BatchedDOMEventType.WHEEL_TYPE || this.type === BatchedDOMEventType.ALT_TYPE ) {
      callback.call( input, this.eventContext );
    }
    else {
      throw new Error( `bad type value: ${this.type}` );
    }

    sceneryLog && sceneryLog.InputEvent && sceneryLog.pop();
  }

  /**
   * Releases references
   */
  public dispose(): void {
    // clear our references
    this.eventContext = null;
    this.callback = null;
    this.freeToPool();
  }

  public freeToPool(): void {
    BatchedDOMEvent.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( BatchedDOMEvent );
}

scenery.register( 'BatchedDOMEvent', BatchedDOMEvent );
