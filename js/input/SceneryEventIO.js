// Copyright 2018-2020, University of Colorado Boulder

/**
 * IO Type for SceneryEvent
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import validate from '../../../axon/js/validate.js';
import Vector2IO from '../../../dot/js/Vector2IO.js';
import ObjectIO from '../../../tandem/js/types/ObjectIO.js';
import scenery from '../scenery.js';
import SceneryEvent from './SceneryEvent.js';

class SceneryEventIO extends ObjectIO {

  /**
   * @param {SceneryEvent} event
   * @returns {Object}
   * @override
   * @public
   */
  static toStateObject( event ) {
    validate( event, this.validator );

    const eventObject = {
      type: event.type
    };

    if ( event.domEvent ) {
      eventObject.domEventType = event.domEvent.type;
    }
    if ( event.pointer && event.pointer.point ) {
      eventObject.point = Vector2IO.toStateObject( event.pointer.point );
    }

    // Note: If changing the contents of this object, please document it in the public documentation string.
    return eventObject;
  }
}

SceneryEventIO.documentation = 'An event, with a point';
SceneryEventIO.validator = { valueType: SceneryEvent };
SceneryEventIO.typeName = 'SceneryEventIO';
ObjectIO.validateSubtype( SceneryEventIO );

scenery.register( 'SceneryEventIO', SceneryEventIO );
export default SceneryEventIO;