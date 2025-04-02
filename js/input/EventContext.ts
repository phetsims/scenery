// Copyright 2023-2025, University of Colorado Boulder

/**
 * A collection of information about an event and the environment when it was fired
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';
import IOType from '../../../tandem/js/types/IOType.js';
import EventIO from '../input/EventIO.js';
import scenery from '../scenery.js';
import { deserializeDomEvent, serializeDomEvent } from './eventSerialization.js';

export default class EventContext<out DOMEvent extends Event = Event> {

  // Raw DOM InputEvent (TouchEvent, PointerEvent, MouseEvent,...)
  public readonly domEvent: DOMEvent;

  // The document.activeElement when the event was fired
  public readonly activeElement: Element | null;

  public constructor( domEvent: DOMEvent ) {
    this.domEvent = domEvent;
    this.activeElement = document.activeElement;
  }

  public static createSynthetic(): EventContext {
    return new EventContext( new window.Event( 'synthetic' ) );
  }

  /**
   * DOM (Scenery) nodes set dataset.sceneryAllowInput on their container if they don't want preventDefault to be called,
   * or other effects that block input (e.g. setPointerCapture). We search up the tree to detect this.
   */
  public allowsDOMInput(): boolean {
    const target = this.domEvent?.target;


    if ( target instanceof Element ) {
      let element: Node | null = target;

      while ( element ) {
        // For DOM nodes, we can check for a data attribute
        if ( element instanceof HTMLElement && element.dataset?.sceneryAllowInput === 'true' ) {
          return true;
        }

        element = element.parentNode;
      }
    }

    return false;
  }
}

export const EventContextIO = new IOType<IntentionalAny, IntentionalAny>( 'EventContextIO', {
  valueType: EventContext,
  documentation: 'A DOM event and its context',
  toStateObject: eventContext => {
    return {
      domEvent: serializeDomEvent( eventContext.domEvent )

      // Ignores the activeElement, since we don't have a good way of serializing that at this point?
    };
  },
  fromStateObject: stateObject => {
    return new EventContext( deserializeDomEvent( stateObject.domEvent ) );
  },

  // This should remain the same as Input.domEventPropertiesToSerialize (local var). Each key can be null depending on
  // what Event interface is being serialized (which depends on what DOM Event the instance is).
  stateSchema: () => ( {
    domEvent: EventIO

    // Ignores the activeElement, since we don't have a good way of serializing that at this point?
  } )
} );

scenery.register( 'EventContext', EventContext );