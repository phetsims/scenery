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

  return inherit( Object, AriaSpeech, {}, {
    init: function() {
      ariaSpeechDiv = document.createElement( 'div' );
      ariaSpeechDiv.id = 'liveText';
      ariaSpeechDiv.className = 'text';
      ariaSpeechDiv.setAttribute( 'aria-live', 'assertive' );

      //Display:none and visibility:hidden both cause aria TTS to fail (no text comes out) on VoiceOver
      document.body.appendChild( ariaSpeechDiv );

      Input.focusedInstanceProperty.link( function( focusedInstance ) {
        if ( focusedInstance && focusedInstance.node && focusedInstance.node.textDescription ) {
          AriaSpeech.setText( focusedInstance.node.textDescription );
        }
      } );
    },
    setText: function( text ) {
      ariaSpeechDiv.innerHTML = text;
    }
  } );
} );