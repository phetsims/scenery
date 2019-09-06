// Copyright 2018-2019, University of Colorado Boulder

/**
 * IO type for a DOMEvent
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

  class DOMEventIO extends ObjectIO {

    /**
     * Encodes a DOMEvent instance to a state.
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
     * @returns {window.Event}
     */
    static fromStateObject( stateObject ) {
      return scenery.Input.deserializeDomEvent( stateObject );
    }
  }

  DOMEventIO.documentation = 'A DOM Event';
  DOMEventIO.validator = { valueType: window.Event };
  DOMEventIO.typeName = 'DOMEventIO';
  ObjectIO.validateSubtype( DOMEventIO );

  return scenery.register( 'DOMEventIO', DOMEventIO );
} );

