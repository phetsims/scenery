// Copyright 2020, University of Colorado Boulder

/**
 * PhET-iO API type for DragListener.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import ActionAPI from '../../../axon/js/ActionAPI.js';
import ActionIO from '../../../axon/js/ActionIO.js';
import merge from '../../../phet-core/js/merge.js';
import EventType from '../../../tandem/js/EventType.js';
import SceneryEventIO from '../input/SceneryEventIO.js';
import scenery from '../scenery.js';
import PressListenerAPI from './PressListenerAPI.js';

class DragListenerAPI extends PressListenerAPI {

  /**
   * @param {Object} [options]
   */
  constructor( options ) {
    options = merge( {
      dragActionOptions: {
        phetioType: ActionIO( [ SceneryEventIO ] ),
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