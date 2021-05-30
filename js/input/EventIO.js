// Copyright 2018-2020, University of Colorado Boulder

/**
 * IOType for a window.Event
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Chris Klusendorf (PhET Interactive Simulations)
 * @author Sam Reid (PhET Interactive Simulations)
 */

import ArrayIO from '../../../tandem/js/types/ArrayIO.js';
import BooleanIO from '../../../tandem/js/types/BooleanIO.js';
import IOType from '../../../tandem/js/types/IOType.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import NumberIO from '../../../tandem/js/types/NumberIO.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import scenery from '../scenery.js';

const EventIO = new IOType( 'EventIO', {
  valueType: window.Event,
  documentation: 'A DOM Event',
  toStateObject: domEvent => scenery.Input.serializeDomEvent( domEvent ),
  fromStateObject: stateObject => scenery.Input.deserializeDomEvent( stateObject ),

  // This should remain the same as Input.domEventPropertiesToSerialize (local var). Each key can be null depending on
  // what Event interface is being serialized (which depends on what DOM Event the instance is).
  stateSchema: EventIO => ( {
    pointerId: NullableIO( NumberIO ),
    pointerType: NullableIO( StringIO ),
    clientX: NullableIO( NumberIO ),
    clientY: NullableIO( NumberIO ),
    ctrlKey: NullableIO( BooleanIO ),
    shiftKey: NullableIO( BooleanIO ),
    altKey: NullableIO( BooleanIO ),
    metaKey: NullableIO( BooleanIO ),
    button: NullableIO( NumberIO ),
    relatedTarget: NullableIO( EventIO ),
    pageX: NullableIO( NumberIO ),
    pageY: NullableIO( NumberIO ),
    which: NullableIO( NumberIO ),
    type: NullableIO( StringIO ),
    target: NullableIO( EventIO ),
    keyCode: NullableIO( NumberIO ),
    key: NullableIO( StringIO ),
    deltaX: NullableIO( NumberIO ),
    deltaY: NullableIO( NumberIO ),
    deltaZ: NullableIO( NumberIO ),
    deltaMode: NullableIO( NumberIO ),
    charCode: NullableIO( NumberIO ),
    changedTouches: NullableIO( ArrayIO( EventIO ) ),
    scale: NullableIO( NumberIO )
  } )
} );

scenery.register( 'EventIO', EventIO );
export default EventIO;