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
  var Emitter = require( 'AXON/Emitter' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @constructor
   */
  function Reader( cursor ) {

    var self = this;

    // @public, listen only, emits an event when the synth begins speaking the utterance
    this.speakingStartedEmitter = new Emitter( { validationEnabled: false } );

    // @public, listen only, emits an event when the synth has finished speaking the utterance
    this.speakingEndedEmitter = new Emitter( { validationEnabled: false } );

    // @private, flag for when screen reader is speaking - synth.speaking is unsupported for safari
    this.speaking = false;

    // @private, keep track of the polite utterances to assist with the safari specific bug, see below
    self.politeUtterances = [];

    // windows Chrome needs a temporary workaround to avoid skipping every other utterance
    // TODO: Use platform.js and revisit once platforms fix their bugs
    var userAgent = navigator.userAgent;
    var osWindows = userAgent.match( /Windows/ );
    var platSafari = !!( userAgent.match( /Version\/[5-9]\./ ) && userAgent.match( /Safari\// ) && userAgent.match( /AppleWebKit/ ) );

    if ( window.speechSynthesis && SpeechSynthesisUtterance && window.speechSynthesis.speak ) {

      // @private - the speech synth
      this.synth = window.speechSynthesis;

      cursor.outputUtteranceProperty.link( function( outputUtterance ) {

        // create a new utterance
        var utterThis = new SpeechSynthesisUtterance( outputUtterance.text );

        utterThis.addEventListener( 'start', function( event ) {
          self.speakingStartedEmitter.emit( outputUtterance );
        } );

        utterThis.addEventListener( 'end', function( event ) {
          self.speakingEndedEmitter.emit( outputUtterance );
        } );

        // get the default voice
        var defaultVoice;
        self.synth.getVoices().forEach( function( voice ) {
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

          // empty the queue of polite utterances
          self.politeUtterances = [];
          self.speaking = true;

          // if assertive or off, cancel the current active utterance and begin speaking immediately
          // TODO: This is how most screen readers work, but we will probably want different behavior
          // for sims so multiple assertive updates do not compete.
          
          // On Windows, the synth must be paused before cancelation and resumed after speaking, 
          // or every other utterance will be skipped.
          // NOTE: This only seems to happen on Windows for the default voice?
          if ( osWindows ) {
            self.synth.pause();
            self.synth.cancel();
            self.synth.speak( utterThis );
            self.synth.resume();
          }
          else {
            self.synth.cancel();
            self.synth.speak( utterThis );
          }
        }
        else if ( outputUtterance.liveRole === 'polite' ) {

          // handle the safari specific bug where 'end' and 'start' events are fired on all utterances
          // after they are added to the queue
          if ( platSafari ) {
            self.politeUtterances.push( utterThis );

            var readPolite = function() {
              self.speaking = true;
              var nextUtterance = self.politeUtterances.shift();
              if ( nextUtterance ) {
                self.synth.speak( nextUtterance );
              }
              else {
                self.speaking = false;
              }
            };

            // a small delay will allow the utterance to be read in full, even if
            // added after cancel().
            if ( self.speaking ) {
              setTimeout( function() { readPolite(); }, 2000 );
            }
            else {
              self.synth.speak( utterThis );
              // remove from queue
              var index = self.politeUtterances.indexOf( utterThis );
              if ( index > 0 ) {
                self.politeUtterances.splice( index, 1 );
              }
            }
          }
          else {
            // simply add to the queue
            self.synth.speak( utterThis );
          } 
        }
      } );
    }
    else {
      cursor.outputUtteranceProperty.link( function() {
        self.speakingStartedEmitter.emit( { text: 'Sorry! Web Speech API not supported on this platform.' } );
      } );
    }

  }

  scenery.register( 'Reader', Reader );

  return inherit( Object, Reader );

} );
