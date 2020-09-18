// Copyright 2018-2020, University of Colorado Boulder

/**
 * IO Type for SceneryEvent
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import Vector2IO from '../../../dot/js/Vector2IO.js';
import IOType from '../../../tandem/js/types/IOType.js';
import scenery from '../scenery.js';
import SceneryEvent from './SceneryEvent.js';

const SceneryEventIO = new IOType( 'SceneryEventIO', {
  valueType: SceneryEvent,
  documentation: 'An event, with a point',
  toStateObject: event => {

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
} );

scenery.register( 'SceneryEventIO', SceneryEventIO );
export default SceneryEventIO;