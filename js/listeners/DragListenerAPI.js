// Copyright 2020, University of Colorado Boulder

/**
 * PhET-iO API type for DragListener.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import Action from '../../../axon/js/Action.js';
import ActionAPI from '../../../axon/js/ActionAPI.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import SceneryEvent from '../input/SceneryEvent.js';
import scenery from '../scenery.js';
import PressListenerAPI from './PressListenerAPI.js';

class DragListenerAPI extends PressListenerAPI {

  /**
   * @param {Object} [options]
   */
  constructor( options ) {
    options = merge( {
      dragActionOptions: {
        phetioType: Action.ActionIO( [ SceneryEvent.SceneryEventIO ] ),
        phetioEventType: EventType.USER,
        phetioHighFrequency: true,
        phetioReadOnly: true
      }
    }, options );
    super( options );

    this.dragAction = new ActionAPI( options.dragActionOptions );
  }
}

scenery.register( 'DragListenerAPI', DragListenerAPI );
export default DragListenerAPI;