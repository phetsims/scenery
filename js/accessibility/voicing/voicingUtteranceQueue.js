// Copyright 2021-2022, University of Colorado Boulder

/**
 * A singleton UtteranceQueue that is used for Voicing. It uses the voicingManager to announce Utterances,
 * which uses HTML5 SpeechSynthesis. This UtteranceQueue can take special VoicingUtterances, which
 * have some extra functionality for controlling flow of alerts.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import UtteranceQueue from '../../../../utterance-queue/js/UtteranceQueue.js';
import { scenery, voicingManager } from '../../imports.js';

const voicingUtteranceQueue = new UtteranceQueue( voicingManager, {
  featureSpecificAnnouncingControlPropertyName: 'voicingCanAnnounceProperty'
} );

// voicingUtteranceQueue should be disabled until requested
voicingUtteranceQueue.enabled = false;

scenery.register( 'voicingUtteranceQueue', voicingUtteranceQueue );
export default voicingUtteranceQueue;
