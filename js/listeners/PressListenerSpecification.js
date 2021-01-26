// Copyright 2020, University of Colorado Boulder

import Action from '../../../axon/js/Action.js';
import ActionAPI from '../../../axon/js/ActionAPI.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import ObjectSpecification from '../../../tandem/js/ObjectSpecification.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
import SceneryEvent from '../input/SceneryEvent.js';
import scenery from '../scenery.js';

class PressListenerSpecification extends ObjectSpecification {

  /**
   * TODO: https://github.com/phetsims/phet-io/issues/1657 Pass instance instead of getter
   * @param {Object} [options]
   */
  constructor( getParent, childName, options ) {
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

    super( getParent, childName );

    // @public (read-only)
    this.pressAction = new ActionAPI( options.pressActionOptions );
    this.releaseAction = new ActionAPI( options.releaseActionOptions );
  }
}

scenery.register( 'PressListenerSpecification', PressListenerSpecification );
export default PressListenerSpecification;