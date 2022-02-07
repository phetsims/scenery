// Copyright 2020-2022, University of Colorado Boulder

/**
 * Uses the Web Speech API to produce speech from the browser. This is a prototype, DO NOT USE IN PRODUCTION CODE.
 * There is no speech output until the voicingManager has been initialized. Supported voices will depend on platform.
 * For each voice, you can customize the rate and pitch. Only one voicingManager should be active at a time and so this
 * type is a singleton.
 *
 * @author Jesse Greenberg
 */

import SpeechSynthesisAnnouncer from '../../../../utterance-queue/js/SpeechSynthesisAnnouncer.js';
import { globalKeyStateTracker, KeyboardUtils, scenery } from '../../imports.js';

class VoicingManager extends SpeechSynthesisAnnouncer {
  constructor() {
    super( {

      // {boolean} - All VoicingManager instances should respect responseCollector's current state.
      respectResponseCollectorProperties: true
    } );
  }

  /**
   * The initialization with some additional scenery-specific work for voicingManager.
   * @override
   * @public
   *
   * @param {Emitter} userGestureEmitter
   * @param {Object} [options]
   */
  initialize( userGestureEmitter, options ) {
    super.initialize( userGestureEmitter, options );

    // The control key will stop the synth from speaking if there is an active utterance. This key was decided because
    // most major screen readers will stop speech when this key is pressed
    globalKeyStateTracker.keyupEmitter.addListener( domEvent => {
      if ( KeyboardUtils.isControlKey( domEvent ) ) {
        this.cancel();
      }
    } );
  }
}

const voicingManager = new VoicingManager();

scenery.register( 'voicingManager', voicingManager );
export default voicingManager;