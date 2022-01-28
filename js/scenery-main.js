// Copyright 2016-2019, University of Colorado Boulder

import axon from '../../axon/js/main.js'; // eslint-disable-line default-import-match-filename
import dot from '../../dot/js/main.js'; // eslint-disable-line default-import-match-filename
import kite from '../../kite/js/main.js'; // eslint-disable-line default-import-match-filename
import phetCore from '../../phet-core/js/main.js'; // eslint-disable-line default-import-match-filename
import utteranceQueue from '../../utterance-queue/js/main.js'; // eslint-disable-line default-import-match-filename
import scenery from './main.js'; // eslint-disable-line default-import-match-filename

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