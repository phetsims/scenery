// Copyright 2016, University of Colorado Boulder

/**
 * A reader of text content for accessibility.  This takes a Cursor reads its output.  This prototype
 * uses the Web Speech API as a synthesizer for text to speech.
 *
 * See https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API
 * @author Jesse Greenberg
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Emitter = require( 'AXON/Emitter' );

  /**
   * @constructor
   */
  function Reader( cursor ) {

    var thisReader = this;

    // @public, listen only, emits an event when the synth begins speaking the utterance
    this.speakingStartedEmitter = new Emitter();

    // @public, listen only, emits an event when the synth has finished speaking the utterance
    this.speakingEndedEmitter = new Emitter();

    // windows Chrome needs a temporary workaround to avoid skipping every other utterance
    // TODO: Use platform.js and revisit once Chrome fixes bug
    var osWindows = navigator.userAgent.match( /Windows/ );

    if ( window.speechSynthesis && SpeechSynthesisUtterance && window.speechSynthesis.speak ) {
      // Web Speech API looks good for synthesis, run wild!
      this.synth = window.speechSynthesis;

      cursor.outputUtteranceProperty.lazyLink( function( outputUtterance ) {

        // create a new utterance
        var utterThis = new SpeechSynthesisUtterance( outputUtterance.text );

        utterThis.onstart = function( event ) {
          thisReader.speakingStartedEmitter.emit1( outputUtterance );
        };

        utterThis.onend = function( event ) {
          thisReader.speakingEndedEmitter.emit1( outputUtterance );
        };

        // get the default voice
        var defaultVoice;
        thisReader.synth.getVoices().forEach( function( voice ) {
          if ( voice.default ) {
            defaultVoice = voice;
            return;
          }
        } );

        // set the voice, pitch, and rate for the utterance
        utterThis.voice = defaultVoice;
        utterThis.rate = 1.2;

        // TODO: Implement behavior for the various live roles
        // see https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
        if( outputUtterance.liveRole === 'assertive' ||
            outputUtterance.liveRole === 'off' ||
            !outputUtterance.liveRole ) {

          // if assertive or off, cancel the current active utterance and begin speaking immediately
          // TODO: This is how most screen readers work, but we will probably want different behavior
          // for sims so multiple assertive updates do not compete.
          
          // On Windows, the synth must be paused before cancelation and resumed after speaking, 
          // or every other utterance will be skipped.
          // NOTE: This only seems to happen on Windows for the default voice?
          if ( osWindows ) {
            thisReader.synth.pause();
            thisReader.synth.cancel();
            thisReader.synth.speak( utterThis );
            thisReader.synth.resume();
          }
          else {
            thisReader.synth.cancel();
            thisReader.synth.speak( utterThis );
          }
          thisReader.activeUtterance = utterThis;
        }
        else if ( outputUtterance.liveRole === 'polite' ) {
          // if polite, simply add the live update text to the queue of utterances
          thisReader.synth.speak( utterThis );
        }
      } );
    }
    else {
      console.error( 'This browser that does not support the Web Speech API.' );
    }

  }

  scenery.register( 'Reader', Reader );

  return inherit( Object, Reader );

} );
