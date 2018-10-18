// Copyright 2016, University of Colorado Boulder

/**
 * IO type for scenery Event
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  const Vector2IO = require( 'DOT/Vector2IO' );
  const scenery = require( 'SCENERY/scenery' );

  // ifphetio
  const assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );
  const ObjectIO = require( 'TANDEM/types/ObjectIO' );
  const phetioInherit = require( 'ifphetio!PHET_IO/phetioInherit' );

  /**
   * IO type for phet/scenery's Event class.
   * @param {Event} event
   * @param phetioID
   * @constructor
   */
  function EventIO( event, phetioID ) {
    assert && assertInstanceOf( event, scenery.Event );
    ObjectIO.call( this, event, phetioID );
  }

  phetioInherit( ObjectIO, 'EventIO', EventIO, {}, {
    get documentation() { return 'An event, with a point'; },

    /**
     * Encodes a Color into a state object.
     * @param {Event} event
     * @returns {Object}
     * @override
     */
    toStateObject( event ) {
      assert && assertInstanceOf( event, scenery.Event );

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
  } );

  return scenery.register( 'EventIO', EventIO );
} );

