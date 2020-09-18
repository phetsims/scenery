// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO Type for SCENERY ButtonListener (not SUN ButtonListener)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */

import IOType from '../../../tandem/js/types/IOType.js';
import scenery from '../scenery.js';

const ButtonListenerIO = new IOType( 'ButtonListenerIO', {
  valueType: scenery.ButtonListener,
  documentation: 'Button listener',
  events: [ 'up', 'over', 'down', 'out', 'fire' ]
} );

scenery.register( 'ButtonListenerIO', ButtonListenerIO );
export default ButtonListenerIO;