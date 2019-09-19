// Copyright 2018-2019, University of Colorado Boulder

/**
 * IO type for scenery Event
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  const ObjectIO = require( 'TANDEM/types/ObjectIO' );
  const scenery = require( 'SCENERY/scenery' );
  const Vector2IO = require( 'DOT/Vector2IO' );
  const Event = require( 'SCENERY/input/Event' );
  const validate = require( 'AXON/validate' );

  class EventIO extends ObjectIO {

    /**
     * @param {Event} event
     * @returns {Object}
     * @override
     */
    static toStateObject( event ) {
      validate( event, this.validator );

      var eventObject = {
        type: event.type,
        domEventType: event.domEvent.type
      };
      if ( event.pointer && event.pointer.point ) {
        eventObject.point = Vector2IO.toStateObject( event.pointer.point );
      }

      // Note: If changing the contents of this object, please document it in the public documentation string.
      return eventObject;
    }
  }

  EventIO.documentation = 'An event, with a point';
  EventIO.validator = { valueType: Event };
  EventIO.typeName = 'EventIO';
  ObjectIO.validateSubtype( EventIO );

  return scenery.register( 'EventIO', EventIO );
} );

