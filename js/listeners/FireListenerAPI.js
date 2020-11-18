// Copyright 2020, University of Colorado Boulder

/**
 * PhET-iO API type for FireListener.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import ActionAPI from '../../../axon/js/ActionAPI.js';
import Emitter from '../../../axon/js/Emitter.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import SceneryEvent from '../input/SceneryEvent.js';
import scenery from '../scenery.js';
import PressListenerAPI from './PressListenerAPI.js';

class FireListenerAPI extends PressListenerAPI {

  /**
   * @param {Object} [options]
   */
  constructor( options ) {
    options = merge( {
      firedEmitterOptions: {
        phetioType: Emitter.EmitterIO( [ NullableIO( SceneryEvent.SceneryEventIO ) ] ),
        phetioEventType: EventType.USER
      }
    }, options );

    super( options );

    // @public (read-only)
    this.firedEmitter = new ActionAPI( options.firedEmitterOptions );
  }
}

scenery.register( 'FireListenerAPI', FireListenerAPI );
export default FireListenerAPI;