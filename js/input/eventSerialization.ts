// Copyright 2025, University of Colorado Boulder

/**
 * Serialization and deserialization of DOM events. This is used for recording
 * and playback of events.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import PDOMUtils from '../accessibility/pdom/PDOMUtils.js';

export const TARGET_SUBSTITUTE_KEY = 'targetSubstitute';

// This is the list of keys that get serialized AND deserialized. NOTE: Do not add or change this without
// consulting the PhET-iO IOType schema for this in EventIO
const domEventPropertiesToSerialize = [
  'altKey',
  'button',
  'charCode',
  'clientX',
  'clientY',
  'code',
  'ctrlKey',
  'deltaMode',
  'deltaX',
  'deltaY',
  'deltaZ',
  'key',
  'keyCode',
  'metaKey',
  'pageX',
  'pageY',
  'pointerId',
  'pointerType',
  'scale',
  'shiftKey',
  'target',
  'type',
  'relatedTarget',
  'which'
] as const;

// The list of serialized properties needed for deserialization
type SerializedPropertiesForDeserialization = typeof domEventPropertiesToSerialize[number];

// Cannot be set after construction, and should be provided in the init config to the constructor(), see Input.deserializeDOMEvent
const domEventPropertiesSetInConstructor: SerializedPropertiesForDeserialization[] = [
  'deltaMode',
  'deltaX',
  'deltaY',
  'deltaZ',
  'altKey',
  'button',
  'charCode',
  'clientX',
  'clientY',
  'code',
  'ctrlKey',
  'key',
  'keyCode',
  'metaKey',
  'pageX',
  'pageY',
  'pointerId',
  'pointerType',
  'shiftKey',
  'type',
  'relatedTarget',
  'which'
];

type SerializedDOMEvent = {
  constructorName: string; // used to get the constructor from the window object, see Input.deserializeDOMEvent
} & Partial<Record<SerializedPropertiesForDeserialization, unknown>>;

// A list of keys on events that need to be serialized into HTMLElements
const EVENT_KEY_VALUES_AS_ELEMENTS: SerializedPropertiesForDeserialization[] = [ 'target', 'relatedTarget' ];

/**
 * Saves the main information we care about from a DOM `Event` into a JSON-like structure. To support
 * polymorphism, all supported DOM event keys that scenery uses will always be included in this serialization. If
 * the particular Event interface for the instance being serialized doesn't have a certain property, then it will be
 * set as `null`. See domEventPropertiesToSerialize for the full list of supported Event properties.
 *
 * @returns - see domEventPropertiesToSerialize for list keys that are serialized
 */
export const serializeDomEvent = ( domEvent: Event ): SerializedDOMEvent => {
  const entries: SerializedDOMEvent = {
    constructorName: domEvent.constructor.name
  };

  domEventPropertiesToSerialize.forEach( property => {

    const domEventProperty: Event[ keyof Event ] | Element = domEvent[ property as keyof Event ];

    // We serialize many Event APIs into a single object, so be graceful if properties don't exist.
    if ( domEventProperty === undefined || domEventProperty === null ) {
      entries[ property ] = null;
    }

    else if ( domEventProperty instanceof Element && EVENT_KEY_VALUES_AS_ELEMENTS.includes( property ) && typeof domEventProperty.getAttribute === 'function' &&

              // If false, then this target isn't a PDOM element, so we can skip this serialization
              domEventProperty.hasAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID ) ) {

      // If the target came from the accessibility PDOM, then we want to store the Node trail id of where it came from.
      entries[ property ] = {
        [ PDOMUtils.DATA_PDOM_UNIQUE_ID ]: domEventProperty.getAttribute( PDOMUtils.DATA_PDOM_UNIQUE_ID ),

        // Have the ID also
        id: domEventProperty.getAttribute( 'id' )
      };
    }
    else {

      // Parse to get rid of functions and circular references.
      entries[ property ] = ( ( typeof domEventProperty === 'object' ) ? {} : JSON.parse( JSON.stringify( domEventProperty ) ) );
    }
  } );

  return entries;
};

/**
 * From a serialized dom event, return a recreated window.Event (scenery-internal)
 */
export const deserializeDomEvent = ( eventObject: SerializedDOMEvent ): Event => {
  const constructorName = eventObject.constructorName || 'Event';

  const configForConstructor = _.pick( eventObject, domEventPropertiesSetInConstructor );
  // serialize the relatedTarget back into an event Object, so that it can be passed to the init config in the Event
  // constructor
  if ( configForConstructor.relatedTarget ) {
    // @ts-expect-error
    const htmlElement = document.getElementById( configForConstructor.relatedTarget.id );
    assert && assert( htmlElement, 'cannot deserialize event when related target is not in the DOM.' );
    configForConstructor.relatedTarget = htmlElement;
  }

  // @ts-expect-error
  const domEvent: Event = new window[ constructorName ]( constructorName, configForConstructor );

  for ( const key in eventObject ) {

    // `type` is readonly, so don't try to set it.
    if ( eventObject.hasOwnProperty( key ) && !( domEventPropertiesSetInConstructor as string[] ).includes( key ) ) {

      // Special case for target since we can't set that read-only property. Instead use a substitute key.
      if ( key === 'target' ) {

        if ( assert ) {
          const target = eventObject.target as { id?: string } | undefined;
          if ( target && target.id ) {
            assert( document.getElementById( target.id ), 'target should exist in the PDOM to support playback.' );
          }
        }

        // @ts-expect-error
        domEvent[ TARGET_SUBSTITUTE_KEY ] = _.clone( eventObject[ key ] ) || {};

        // This may not be needed since https://github.com/phetsims/scenery/issues/1296 is complete, double check on getTrailFromPDOMEvent() too
        // @ts-expect-error
        domEvent[ TARGET_SUBSTITUTE_KEY ].getAttribute = function( key ) {
          return this[ key ];
        };
      }
      else {

        // @ts-expect-error
        domEvent[ key ] = eventObject[ key ];
      }
    }
  }
  return domEvent;
};