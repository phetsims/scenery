// Copyright 2020-2022, University of Colorado Boulder

/**
 * A global object that tracks the state of the keyboard for the Window. Use this
 * to get information about which keyboard keys are pressed down and for how long.
 *
 * @author Michael Kauzmann
 * @author Jesse Greenberg
 */

import Tandem from '../../../tandem/js/Tandem.js';
import { KeyStateTracker, scenery } from '../imports.js';
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