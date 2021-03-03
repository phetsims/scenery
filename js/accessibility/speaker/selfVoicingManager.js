// Copyright 2021, University of Colorado Boulder

/**
 * Manages "self-voicing" content as it is read with speech synthesis. Output is categorized into
 *    "Object Responses" - Speech describing the object as it receives interaction
 *    "Context Responses" - Speech describing surrounding contextual changes in response to user interaction
 *    "Hints" - General hint content to guide a particular user interaction
 *
 * Output from each of these categories can be separately enabled and disabled.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import scenery from '../../scenery.js';
import BooleanProperty from '../../../../axon/js/BooleanProperty.js';

class SelfVoicingManager {
  constructor() {

    // @public {BooleanProperty} - whether or not 'object responses' are read as interactive components change
    this.objectChangesProperty = new BooleanProperty( true );

    // @public {BooleanProperty} - whether or not "context responses" are read as simulation objects change
    this.contextChangesProperty = new BooleanProperty( true );

    // @public {BooleanProperty} - whether or not helpful or interaction hints are read to the user
    this.hintsProperty = new BooleanProperty( false );
  }
}

const selfVoicingManager = new SelfVoicingManager();
scenery.register( 'selfVoicingManager', selfVoicingManager );
export default selfVoicingManager;
