// Copyright 2017-2020, University of Colorado Boulder

/**
 * IO type for SCENERY ButtonListener (not SUN ButtonListener)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */

import ObjectIO from '../../../tandem/js/types/ObjectIO.js';
import scenery from '../scenery.js';

class ButtonListenerIO extends ObjectIO {}

ButtonListenerIO.documentation = 'Button listener';
ButtonListenerIO.events = [ 'up', 'over', 'down', 'out', 'fire' ];
ButtonListenerIO.validator = { valueType: scenery.ButtonListener };
ButtonListenerIO.typeName = 'ButtonListenerIO';
ObjectIO.validateSubtype( ButtonListenerIO );

scenery.register( 'ButtonListenerIO', ButtonListenerIO );
export default ButtonListenerIO;