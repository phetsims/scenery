// Copyright 2017-2019, University of Colorado Boulder

/**
 * A type of listener that absorbs all 'down' events, not letting it bubble further to ancestor node listeners.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const inherit = require( 'PHET_CORE/inherit' );
  const scenery = require( 'SCENERY/scenery' );

  /**
   * Creates a listener that absorbs 'down' events, preventing them from bubbling further.
   * @constructor
   *
   * NOTE: This does not call abort(), so listeners that are added to the same Node as this listener will still fire
   *       normally.
   */
  function HandleDownlistener() {
  }

  scenery.register( 'HandleDownlistener', HandleDownlistener );

  inherit( Object, HandleDownlistener, {
    /**
     * Scenery input callback to absorb down events.
     * @public
     *
     * @param {SceneryEvent} event
     */
    down: function( event ) {
      event.handle();
    }
  } );

  return HandleDownlistener;
} );
