// Copyright 2018-2023, University of Colorado Boulder

/**
 * IOType for a window.Event. Since this needs to support any data from any subtype of window.Event, we supply NullableIO
 * attributes for the union of different supported subtypes.  The subtypes are listed at https://developer.mozilla.org/en-US/docs/Web/API/Event
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Chris Klusendorf (PhET Interactive Simulations)
 * @author Sam Reid (PhET Interactive Simulations)
 */

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
  stateSchema: () => ( {
    constructorName: StringIO,
    altKey: NullableIO( BooleanIO ),
    button: NullableIO( NumberIO ),
    charCode: NullableIO( NumberIO ),
    clientX: NullableIO( NumberIO ),
    clientY: NullableIO( NumberIO ),
    code: NullableIO( StringIO ),
    ctrlKey: NullableIO( BooleanIO ),
    deltaMode: NullableIO( NumberIO ),
    deltaX: NullableIO( NumberIO ),
    deltaY: NullableIO( NumberIO ),
    deltaZ: NullableIO( NumberIO ),
    key: NullableIO( StringIO ),
    keyCode: NullableIO( NumberIO ),
    metaKey: NullableIO( BooleanIO ),
    pageX: NullableIO( NumberIO ),
    pageY: NullableIO( NumberIO ),
    pointerId: NullableIO( NumberIO ),
    pointerType: NullableIO( StringIO ),
    scale: NullableIO( NumberIO ),
    shiftKey: NullableIO( BooleanIO ),
    target: NullableIO( ObjectLiteralIO ),
    type: NullableIO( StringIO ),
    relatedTarget: NullableIO( ObjectLiteralIO ),
    which: NullableIO( NumberIO )
  } )
} );

scenery.register( 'EventIO', EventIO );
export default EventIO;