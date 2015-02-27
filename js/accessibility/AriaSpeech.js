//  Copyright 2002-2014, University of Colorado Boulder

/**
 *
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Input = require( 'SCENERY/input/Input' );

  function AriaSpeech() {}

  var ariaSpeechDiv = null;
  var initialized = false;

  return inherit( Object, AriaSpeech, {}, {
    init: function() {
      if ( !initialized ) {
        ariaSpeechDiv = document.createElement( 'div' );
        ariaSpeechDiv.id = 'liveText';
        ariaSpeechDiv.className = 'text';
        ariaSpeechDiv.setAttribute( 'aria-live', 'assertive' );

        //Display:none and visibility:hidden both cause aria TTS to fail (no text comes out) on VoiceOver
        document.body.appendChild( ariaSpeechDiv );

        Input.focusedTrailProperty.link( function( focusedTrail ) {
          if ( focusedTrail && focusedTrail.lastNode().textDescription ) {
            AriaSpeech.setText( focusedTrail.lastNode().textDescription );
          }
        } );
        initialized = true;
      }
    },
    setText: function( text ) {
      if ( !initialized ) {
        AriaSpeech.init();
      }
      ariaSpeechDiv.innerHTML = text;
    }
  } );
} );