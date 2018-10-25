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
  const scenery = require( 'SCENERY/scenery' );
  const phetioInherit = require( 'TANDEM/phetioInherit' );

  // ifphetio
  const assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );

  /**
   * IO type for phet/sun's DOMEvent class.
   * @param {DOMEvent} domEvent
   * @param {string} phetioID
   * @constructor
   */
  function DOMEventIO( domEvent, phetioID ) {
    assert && assertInstanceOf( domEvent, Event ); // Event is the browser DOM event type, not the scenery one.
    ObjectIO.call( this, domEvent, phetioID );
  }

  // we must use phetioInherit, even in es6
  phetioInherit( ObjectIO, 'DOMEventIO', DOMEventIO, {}, {
    documentation: 'A DOM Event',

    /**
     * Encodes a DOMEvent instance to a state.
     * @param {Event} domEvent
     * @returns {Object} - a state object
     * @override
     */
    toStateObject( domEvent ) {
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

