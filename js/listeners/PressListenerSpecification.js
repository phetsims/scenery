// Copyright 2020, University of Colorado Boulder

import Action from '../../../axon/js/Action.js';
import ActionSpecification from '../../../axon/js/ActionSpecification.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import ObjectSpecification from '../../../tandem/js/ObjectSpecification.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import SceneryEvent from '../input/SceneryEvent.js';
import scenery from '../scenery.js';

class PressListenerSpecification extends ObjectSpecification {

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

    super( options );

    // @public (read-only)
    this.pressAction = new ActionSpecification( options.pressActionOptions );
    this.releaseAction = new ActionSpecification( options.releaseActionOptions );
  }

  // @public
  test( pressListener ) {
    super.test( pressListener );
    // TODO: https://github.com/phetsims/phet-io/issues/1657 look up the Actions via phetioEngine and run test on them.
  }
}

scenery.register( 'PressListenerSpecification', PressListenerSpecification );
export default PressListenerSpecification;