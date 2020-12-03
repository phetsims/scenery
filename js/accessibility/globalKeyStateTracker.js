// Copyright 2020, University of Colorado Boulder

/**
 * A global object that tracks the state of the keyboard for the Window. Use this
 * to get information about which keyboard keys are pressed down and for how long.
 *
 * @author Michael Kauzmann
 * @author Jesse Greenberg
 */

import Tandem from '../../../tandem/js/Tandem.js';
import scenery from '../scenery.js';
import KeyStateTracker from './KeyStateTracker.js';

class GlobalKeyStateTracker extends KeyStateTracker {
  constructor( options ) {
    super( options );
    this.attachToWindow();
  }
}

// @public (read-only) {KeyStateTracker} -
const globalKeyStateTracker = new GlobalKeyStateTracker( {
  tandem: Tandem.GENERAL_CONTROLLER.createTandem( 'keyStateTracker' )
} );

scenery.register( 'globalKeyStateTracker', globalKeyStateTracker );
export default globalKeyStateTracker;