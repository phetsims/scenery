import axon from '../../axon/js/main.js';
import dot from '../../dot/js/main.js';
import kite from '../../kite/js/main.js';
import phetCore from '../../phet-core/js/main.js';
import utteranceQueue from '../../utterance-queue/js/main.js';
import scenery from './main.js';

// Copyright 2016-2019, University of Colorado Boulder

if ( !window.hasOwnProperty( '_' ) ) {
  throw new Error( 'Underscore/Lodash not found: _' );
}
if ( !window.hasOwnProperty( '$' ) ) {
  throw new Error( 'jQuery not found: $' );
}


window.axon = axon;
window.dot = dot;
window.kite = kite;
window.phetCore = phetCore;
window.utteranceQueue = utteranceQueue;
window.scenery = scenery;
window.scenery.Utils.polyfillRequestAnimationFrame();