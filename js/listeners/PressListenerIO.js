// Copyright 2020, University of Colorado Boulder

/**
 * IO type for PressListener
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import ActionIO from '../../../axon/js/ActionIO.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import ObjectIO from '../../../tandem/js/types/ObjectIO.js';
import SceneryEventIO from '../input/SceneryEventIO.js';
import scenery from '../scenery.js';

class PressListenerIO extends ObjectIO {}

PressListenerIO.validator = { valueType: scenery.PressListener };

PressListenerIO.documentation = 'IO Type for PressListener';
PressListenerIO.typeName = 'PressListenerIO';

PressListenerIO.api = {
  pressAction: { phetioType: ActionIO( [ SceneryEventIO ] ) },
  releaseAction: { phetioType: ActionIO( [ NullableIO( SceneryEventIO ) ] ) }
};

// This IOType won't be instantiated, but instead is used for its api.
PressListenerIO.uninstrumented = true;

ObjectIO.validateSubtype( PressListenerIO );

scenery.register( 'PressListenerIO', PressListenerIO );
export default PressListenerIO;