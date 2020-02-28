// Copyright 2018-2020, University of Colorado Boulder

/**
 * IO type for a window.Event
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Chris Klusendorf (PhET Interactive Simulations)
 * @author Sam Reid (PhET Interactive Simulations)
 */

import validate from '../../../axon/js/validate.js';
import ObjectIO from '../../../tandem/js/types/ObjectIO.js';
import scenery from '../scenery.js';

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

scenery.register( 'EventIO', EventIO );
export default EventIO;