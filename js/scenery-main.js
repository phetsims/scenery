// Copyright 2016-2019, University of Colorado Boulder
( function() {
  'use strict';
  if ( !window.hasOwnProperty( '_' ) ) {
    throw new Error( 'Underscore/Lodash not found: _' );
  }
  if ( !window.hasOwnProperty( '$' ) ) {
    throw new Error( 'jQuery not found: $' );
  }
  define( require => {

    window.axon = require( 'AXON/main' );
    window.dot = require( 'DOT/main' );
    window.kite = require( 'KITE/main' );
    window.phetCore = require( 'PHET_CORE/main' );
    window.utteranceQueue = require( 'UTTERANCE_QUEUE/main' );
    window.scenery = require( 'main' );
    window.scenery.Util.polyfillRequestAnimationFrame();
  } );
} )();