// Copyright 2013-2023, University of Colorado Boulder

/**
 * A Scenery event is an abstraction over incoming user DOM events.
 *
 * It provides more information (particularly Scenery-related information), and handles a single pointer at a time
 * (DOM TouchEvents can include information for multiple touches at the same time, so the TouchEvent can be passed to
 * multiple Scenery events). Thus it is not save to assume that the DOM event is unique, as it may be shared.
 *
 * NOTE: While the event is being dispatched, its currentTarget may be changed. It is not fully immutable.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import IOType from '../../../tandem/js/types/IOType.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import { EventContext, Mouse, Node, PDOMPointer, Pointer, scenery, Trail } from '../imports.js';
import EventIO from './EventIO.js';

// "out" here ensures that SceneryListenerFunctions don't specify a wider type arguments for the event, see  https://github.com/phetsims/scenery/issues/1483
export default class SceneryEvent<out DOMEvent extends Event = Event> {

  // Whether this SceneryEvent has been 'handled'. If so, it will not bubble further.
  public handled: boolean;

  // Whether this SceneryEvent has been 'aborted'. If so, no further listeners with it will fire.
  public aborted: boolean;

  // Path to the leaf-most node "hit" by the event, ordered list, from root to leaf
  public readonly trail: Trail;

  // What event was triggered on the listener, e.g. 'move'
  public readonly type: string;

  // The pointer that triggered this event
  public readonly pointer: Pointer;

  // Raw DOM InputEvent (TouchEvent, PointerEvent, MouseEvent,...)
  public readonly domEvent: DOMEvent | null;

  // Assorted environment information when the event was fired
  public readonly context: EventContext;

  // The document.activeElement when the event was fired
  public readonly activeElement: Element | null;

  // Whatever node you attached the listener to, or null when firing events on a Pointer
  public currentTarget: Node | null;

  // Leaf-most node in trail
  public target: Node;

  // Whether this is the 'primary' mode for the pointer. Always true for touches, and will be true
  // for the mouse if it is the primary (left) mouse button.
  public isPrimary: boolean;

  /**
   * @param trail - The trail to the node picked/hit by this input event.
   * @param type - Type of the event, e.g. 'string'
   * @param pointer - The pointer that triggered this event
   * @param context - The original DOM EventContext that caused this SceneryEvent to fire.
   */
  public constructor( trail: Trail, type: string, pointer: Pointer, context: EventContext<DOMEvent> ) {
    // TODO: add domEvent type assertion -- will browsers support this?

    this.handled = false;
    this.aborted = false;
    this.trail = trail;
    this.type = type;
    this.pointer = pointer;
    this.context = context;
    this.domEvent = context.domEvent;
    this.activeElement = context.activeElement;
    this.currentTarget = null;
    this.target = trail.lastNode();

    // TODO: don't require check on domEvent (seems sometimes this is passed as null as a hack?)
    this.isPrimary = !( pointer instanceof Mouse ) || !this.domEvent || ( this.domEvent as unknown as MouseEvent ).button === 0;

    // Store the last-used non-null DOM event for future use if required.
    pointer.lastEventContext = context;
  }

  /**
   * like DOM SceneryEvent.stopPropagation(), but named differently to indicate it doesn't fire that behavior on the underlying DOM event
   */
  public handle(): void {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'handled event' );
    this.handled = true;
  }

  /**
   * like DOM SceneryEvent.stopImmediatePropagation(), but named differently to indicate it doesn't fire that behavior on the underlying DOM event
   */
  public abort(): void {
    sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'aborted event' );
    this.aborted = true;
  }

  /**
   * Specifies whether the SceneryEvent came from alternative input. See Input.PDOM_EVENT_TYPES for a list of events
   * pdom-related events supported by scenery. These events are exclusively supported by the ParallelDOM for Interactive
   * description.
   */
  public isFromPDOM(): boolean {
    return this.pointer instanceof PDOMPointer;
  }

  /**
   * Returns whether a typical PressListener (that isn't already attached) could start a drag with this event.
   *
   * This can typically be used for patterns where no action should be taken if a press can't be started, e.g.:
   *
   *   down: function( event ) {
   *     if ( !event.canStartPress() ) { return; }
   *
   *     // ... Do stuff to create a node with some type of PressListener
   *
   *     dragListener.press( event );
   *   }
   *
   * NOTE: This ignores non-left mouse buttons (as this is the typical behavior). Custom checks should be done if this
   *       is not suitable.
   */
  public canStartPress(): boolean {
    // If the pointer is already attached (some other press probably), it can't start a press.
    // Additionally, we generally want to ignore non-left mouse buttons.
    return !this.pointer.isAttached() && ( !( this.pointer instanceof Mouse ) || ( this.domEvent as unknown as MouseEvent ).button === 0 );
  }

  public static readonly SceneryEventIO = new IOType( 'SceneryEventIO', {
    valueType: SceneryEvent,
    documentation: 'An event, with a "point" field',
    toStateObject: ( event: SceneryEvent ) => {

      // Note: If changing the contents of this object, please document it in the public documentation string.
      return {
        type: event.type,
        domEventType: NullableIO( EventIO ).toStateObject( event.domEvent ),
        point: ( event.pointer && event.pointer.point ) ? Vector2.Vector2IO.toStateObject( event.pointer.point ) : null
      };
    },
    stateSchema: {
      type: StringIO,
      domEventType: NullableIO( EventIO ),
      point: NullableIO( Vector2.Vector2IO )
    }
  } );

}


scenery.register( 'SceneryEvent', SceneryEvent );
