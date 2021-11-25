// Copyright 2016-2021, University of Colorado Boulder

/**
 * A reader of text content for accessibility.  This takes a Cursor reads its output.  This prototype
 * uses the Web Speech API as a synthesizer for text to speech.
 *
 * See https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API
 *
 * NOTE: We are no longer actively developing this since we know that users would much rather use their own
 * dedicated software. But we are keeping it around for when we want to explore any other voicing features
 * using the web speech API.
 * @author Jesse Greenberg
 */

import Emitter from '../../../../axon/js/Emitter.js';
import { scenery } from '../../imports.js';

class Reader {
  /**
   * @param {Cursor} cursor
   */
  constructor( cursor ) {

    // @public, listen only, emits an event when the synth begins speaking the utterance
    this.speakingStartedEmitter = new Emitter( { parameters: [ { valueType: Object } ] } );

    // @public, listen only, emits an event when the synth has finished speaking the utterance
    this.speakingEndedEmitter = new Emitter( { parameters: [ { valueType: Object } ] } );

    // @private, flag for when screen reader is speaking - synth.speaking is unsupported for safari
    this.speaking = false;

    // @private, keep track of the polite utterances to assist with the safari specific bug, see below
    this.politeUtterances = [];

    // windows Chrome needs a temporary workaround to avoid skipping every other utterance
    // TODO: Use platform.js and revisit once platforms fix their bugs
    const userAgent = navigator.userAgent;
    const osWindows = userAgent.match( /Windows/ );
    const platSafari = !!( userAgent.match( /Version\/[5-9]\./ ) && userAgent.match( /Safari\// ) && userAgent.match( /AppleWebKit/ ) );

    if ( window.speechSynthesis && SpeechSynthesisUtterance && window.speechSynthesis.speak ) {

      // @private - the speech synth
      this.synth = window.speechSynthesis;

      cursor.outputUtteranceProperty.link( outputUtterance => {

        // create a new utterance
        const utterThis = new SpeechSynthesisUtterance( outputUtterance.text );

        utterThis.addEventListener( 'start', event => {
          this.speakingStartedEmitter.emit( outputUtterance );
        } );

        utterThis.addEventListener( 'end', event => {
          this.speakingEndedEmitter.emit( outputUtterance );
        } );

        // get the default voice
        let defaultVoice;
        this.synth.getVoices().forEach( voice => {
          if ( voice.default ) {
            defaultVoice = voice;
          }
        } );

        // set the voice, pitch, and rate for the utterance
        utterThis.voice = defaultVoice;
        utterThis.rate = 1.2;

        // TODO: Implement behavior for the various live roles
        // see https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
        if ( outputUtterance.liveRole === 'assertive' ||
             outputUtterance.liveRole === 'off' ||
             !outputUtterance.liveRole ) {

          // empty the queue of polite utterances
          this.politeUtterances = [];
          this.speaking = true;

          // if assertive or off, cancel the current active utterance and begin speaking immediately
          // TODO: This is how most screen readers work, but we will probably want different behavior
          // for sims so multiple assertive updates do not compete.

          // On Windows, the synth must be paused before cancelation and resumed after speaking,
          // or every other utterance will be skipped.
          // NOTE: This only seems to happen on Windows for the default voice?
          if ( osWindows ) {
            this.synth.pause();
            this.synth.cancel();
            this.synth.speak( utterThis );
            this.synth.resume();
          }
          else {
            this.synth.cancel();
            this.synth.speak( utterThis );
          }
        }
        else if ( outputUtterance.liveRole === 'polite' ) {

          // handle the safari specific bug where 'end' and 'start' events are fired on all utterances
          // after they are added to the queue
          if ( platSafari ) {
            this.politeUtterances.push( utterThis );

            const readPolite = () => {
              this.speaking = true;
              const nextUtterance = this.politeUtterances.shift();
              if ( nextUtterance ) {
                this.synth.speak( nextUtterance );
              }
              else {
                this.speaking = false;
              }
            };

            // a small delay will allow the utterance to be read in full, even if
            // added after cancel().
            if ( this.speaking ) {
              setTimeout( () => { readPolite(); }, 2000 ); // eslint-disable-line bad-sim-text
            }
            else {
              this.synth.speak( utterThis );
              // remove from queue
              const index = this.politeUtterances.indexOf( utterThis );
              if ( index > 0 ) {
                this.politeUtterances.splice( index, 1 );
              }
            }
          }
          else {
            // simply add to the queue
            this.synth.speak( utterThis );
          }
        }
      } );
    }
    else {
      cursor.outputUtteranceProperty.link( () => {
        this.speakingStartedEmitter.emit( { text: 'Sorry! Web Speech API not supported on this platform.' } );
      } );
    }
  }
}

scenery.register( 'Reader', Reader );
export default Reader;