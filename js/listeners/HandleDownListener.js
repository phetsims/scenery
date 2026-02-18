// Copyright 2017-2026, University of Colorado Boulder

/**
 * A type of listener that absorbs all 'down' events, not letting it bubble further to ancestor node listeners.
 *
 * NOTE: This does not call abort(), so listeners that are added to the same Node as this listener will still fire
 *       normally.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import scenery from '../scenery.js';

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