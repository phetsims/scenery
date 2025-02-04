// Copyright 2020-2025, University of Colorado Boulder

/**
 * A global object that tracks the state of the keyboard for the Window. Use this
 * to get information about which keyboard keys are pressed down and for how long.
 *
 * @author Michael Kauzmann
 * @author Jesse Greenberg
 */

import Tandem from '../../../tandem/js/Tandem.js';
import KeyStateTracker from '../accessibility/KeyStateTracker.js';
import scenery from '../scenery.js';
import { KeyStateTrackerOptions } from './KeyStateTracker.js';

class GlobalKeyStateTracker extends KeyStateTracker {
  public constructor( options?: KeyStateTrackerOptions ) {
    super( options );
  }
}

const globalKeyStateTracker = new GlobalKeyStateTracker( {
  tandem: Tandem.GENERAL_CONTROLLER.createTandem( 'keyStateTracker' )
} );

scenery.register( 'globalKeyStateTracker', globalKeyStateTracker );
export default globalKeyStateTracker;