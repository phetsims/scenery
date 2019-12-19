// Copyright 2018-2019, University of Colorado Boulder

/**
 * IO type for a window.Event
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Chris Klusendorf (PhET Interactive Simulations)
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  const ObjectIO = require( 'TANDEM/types/ObjectIO' );
  const scenery = require( 'SCENERY/scenery' );
  const validate = require( 'AXON/validate' );

  class EventIO extends ObjectIO {

    /**
     * Encodes an Event instance to a state.
     * @param {Event} domEvent
     * @returns {Object} - a state object
     * @override
     */
    static toStateObject( domEvent ) {
      validate( domEvent, this.validator );
      return scenery.Input.serializeDomEvent( domEvent );
    }

    /**
     * @param {Object} stateObject
     * @returns {Event}
     */
    static fromStateObject( stateObject ) {
      return scenery.Input.deserializeDomEvent( stateObject );
    }
  }

  EventIO.documentation = 'A DOM Event';
  EventIO.validator = { valueType: window.Event };
  EventIO.typeName = 'EventIO';
  ObjectIO.validateSubtype( EventIO );

  return scenery.register( 'EventIO', EventIO );
} );

