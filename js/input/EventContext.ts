// Copyright 2023, University of Colorado Boulder

/**
 * A collection of information about an event and the environment when it was fired
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import IOType from '../../../tandem/js/types/IOType.js';
import { scenery, EventIO, Input } from '../imports.js';

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
}

export const EventContextIO = new IOType( 'EventContextIO', {
  valueType: EventContext,
  documentation: 'A DOM event and its context',
  toStateObject: eventContext => {
    return {
      domEvent: Input.serializeDomEvent( eventContext.domEvent )

      // Ignores the activeElement, since we don't have a good way of serializing that at this point?
    };
  },
  fromStateObject: stateObject => {
    return new EventContext( Input.deserializeDomEvent( stateObject.domEvent ) );
  },

  // This should remain the same as Input.domEventPropertiesToSerialize (local var). Each key can be null depending on
  // what Event interface is being serialized (which depends on what DOM Event the instance is).
  stateSchema: () => ( {
    domEvent: EventIO

    // Ignores the activeElement, since we don't have a good way of serializing that at this point?
  } )
} );

scenery.register( 'EventContext', EventContext );
