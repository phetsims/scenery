// Copyright 2020, University of Colorado Boulder

/**
 * PhET-iO API type for PressListener. Since PressListener is not an instrumented PhetioObject, but rather holds
 * instrumented sub-components, this API does not extend PhetioObjectAPI.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import Action from '../../../axon/js/Action.js';
import ActionAPI from '../../../axon/js/ActionAPI.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import UninstrumentedAPI from '../../../tandem/js/UninstrumentedAPI.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import SceneryEvent from '../input/SceneryEvent.js';
import scenery from '../scenery.js';

class PressListenerAPI extends UninstrumentedAPI {

  /**
   * @param {Object} [options]
   */
  constructor( options ) {
    options = merge( {
      pressActionOptions: {
        phetioType: Action.ActionIO( [ SceneryEvent.SceneryEventIO ] ),
        phetioEventType: EventType.USER,
        phetioReadOnly: true
      },
      releaseActionOptions: {
        phetioType: Action.ActionIO( [ NullableIO( SceneryEvent.SceneryEventIO ) ] ),
        phetioEventType: EventType.USER,
        phetioReadOnly: true
      }
    }, options );

    super();

    // @public (read-only)
    this.pressAction = new ActionAPI( options.pressActionOptions );
    this.releaseAction = new ActionAPI( options.releaseActionOptions );
  }
}

scenery.register( 'PressListenerAPI', PressListenerAPI );
export default PressListenerAPI;