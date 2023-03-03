// Copyright 2020-2023, University of Colorado Boulder

/**
 * Uses the Web Speech API to produce speech from the browser. This is a prototype, DO NOT USE IN PRODUCTION CODE.
 * There is no speech output until the voicingManager has been initialized. Supported voices will depend on platform.
 * For each voice, you can customize the rate and pitch. Only one voicingManager should be active at a time and so this
 * type is a singleton.
 *
 * @author Jesse Greenberg
 */

import SpeechSynthesisAnnouncer, { SpeechSynthesisAnnouncerOptions, SpeechSynthesisInitializeOptions } from '../../../../utterance-queue/js/SpeechSynthesisAnnouncer.js';
import Tandem from '../../../../tandem/js/Tandem.js';
import { globalKeyStateTracker, KeyboardUtils, scenery } from '../../imports.js';
import optionize, { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import TEmitter from '../../../../axon/js/TEmitter.js';

type SelfOptions = EmptySelfOptions;
type VoicingManagerOptions = SelfOptions & SpeechSynthesisAnnouncerOptions;


class VoicingManager extends SpeechSynthesisAnnouncer {
  public constructor( providedOptions?: VoicingManagerOptions ) {

    const options = optionize<VoicingManagerOptions, SelfOptions, SpeechSynthesisAnnouncerOptions>()( {

      // All VoicingManager instances should respect responseCollector's current state.
      respectResponseCollectorProperties: true,

      // phet-io
      tandem: Tandem.OPTIONAL,
      phetioDocumentation: 'Announcer that manages the voicing feature, providing audio responses via WebAudio.'
    }, providedOptions );

    super( options );
  }

  /**
   * The initialization with some additional scenery-specific work for voicingManager.
   */
  public override initialize( userGestureEmitter: TEmitter, options?: SpeechSynthesisInitializeOptions ): void {
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