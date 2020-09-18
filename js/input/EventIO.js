// Copyright 2018-2020, University of Colorado Boulder

/**
 * IO Type for a window.Event
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Chris Klusendorf (PhET Interactive Simulations)
 * @author Sam Reid (PhET Interactive Simulations)
 */

import IOType from '../../../tandem/js/types/IOType.js';
import scenery from '../scenery.js';

const EventIO = new IOType( 'EventIO', {
  valueType: window.Event,
  documentation: 'A DOM Event',
  toStateObject: domEvent => scenery.Input.serializeDomEvent( domEvent ),
  fromStateObject: stateObject => scenery.Input.deserializeDomEvent( stateObject )
} );

scenery.register( 'EventIO', EventIO );
export default EventIO;