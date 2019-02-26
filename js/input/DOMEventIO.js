// Copyright 2018, University of Colorado Boulder

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
  const phetioInherit = require( 'TANDEM/phetioInherit' );
  const scenery = require( 'SCENERY/scenery' );
  const validate = require( 'AXON/validate' );

  /**
   * IO type for phet/sun's DOMEvent class.
   * @param {DOMEvent} domEvent
   * @param {string} phetioID
   * @constructor
   */
  function DOMEventIO( domEvent, phetioID ) {
    ObjectIO.call( this, domEvent, phetioID );
  }

  // we must use phetioInherit, even in es6
  phetioInherit( ObjectIO, 'DOMEventIO', DOMEventIO, {}, {
    documentation: 'A DOM Event',

    /**
     * Note this is a DOM event, not a scenery.Event
     * @override
     * @public
     */
    validator: { valueType: window.Event },

    /**
     * Encodes a DOMEvent instance to a state.
     * @param {Event} domEvent
     * @returns {Object} - a state object
     * @override
     */
    toStateObject( domEvent ) {
      validate( domEvent, this.validator );
      return scenery.Input.serializeDomEvent( domEvent );
    },

    /**
     * @param {Object} stateObject
     * @returns {window.Event}
     */
    fromStateObject( stateObject ) {
      return scenery.Input.deserializeDomEvent( stateObject );
    }
  } );

  return scenery.register( 'DOMEventIO', DOMEventIO );
} );

