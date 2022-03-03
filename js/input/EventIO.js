// Copyright 2018-2021, University of Colorado Boulder

/**
 * IOType for a window.Event. Since this needs to support any data from any subtype of window.Event, we supply NullableIO
 * attributes for the union of different supported subtypes.  The subtypes are listed at https://developer.mozilla.org/en-US/docs/Web/API/Event
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
import ObjectLiteralIO from '../../../tandem/js/types/ObjectLiteralIO.js';
import StringIO from '../../../tandem/js/types/StringIO.js';
import { Input, scenery } from '../imports.js';

const EventIO = new IOType( 'EventIO', {
  valueType: window.Event,
  documentation: 'A DOM Event',
  toStateObject: domEvent => Input.serializeDomEvent( domEvent ),
  fromStateObject: stateObject => Input.deserializeDomEvent( stateObject ),

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
    relatedTarget: NullableIO( ObjectLiteralIO ),
    pageX: NullableIO( NumberIO ),
    pageY: NullableIO( NumberIO ),
    which: NullableIO( NumberIO ),
    type: NullableIO( StringIO ),
    target: NullableIO( ObjectLiteralIO ),
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