// Copyright 2017-2021, University of Colorado Boulder

/**
 * A type of listener that absorbs all 'down' events, not letting it bubble further to ancestor node listeners.
 *
 * NOTE: This does not call abort(), so listeners that are added to the same Node as this listener will still fire
 *       normally.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../imports.js';

class HandleDownlistener {
  /**
   * Scenery input callback to absorb down events.
   * @public
   *
   * @param {SceneryEvent} event
   */
  down( event ) {
    event.handle();
  }
}

scenery.register( 'HandleDownlistener', HandleDownlistener );
export default HandleDownlistener;