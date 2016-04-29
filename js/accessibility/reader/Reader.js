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

  /**
   * @constructor
   */
  function Reader( cursor ) {

    var thisReader = this;

    if ( window.speechSynthesis && SpeechSynthesisUtterance && window.speechSynthesis.speak ) {
      // Web Speech API looks good for synthesis, run wild!
      this.synth = window.speechSynthesis;

      cursor.outputUtteranceProperty.lazyLink( function( outputUtterance ) {

        // create a new utterance
        var utterThis = new SpeechSynthesisUtterance( outputUtterance.text );

        // set the voice, pitch, and rate for the utterance
        utterThis.voice = thisReader.synth.getVoices()[ 2 ];
        utterThis.pitch = 0.8;
        utterThis.rate = 1.0;

        // TODO: Implement behavior for the various live roles
        // see https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions
        if( outputUtterance.liveRole === 'assertive' ||
            outputUtterance.liveRole === 'off' ||
            !outputUtterance.liveRole ) {

          // if assertive or off, cancel the current active utterance and begin speaking immediately
          // TODO: This is how most screen readers work, but we will probably want different behavior
          // for sims so multiple assertive updates do not compete.
          
          // for some reason, the synth must be paused before cancelation, or every other utterance
          // will be skipped - is this a bug with the API or am I forgetting something?
          thisReader.synth.pause();
          thisReader.synth.cancel();

          thisReader.synth.speak( utterThis );
          thisReader.synth.resume();
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

  return inherit( Object, Reader, {} );

} );
