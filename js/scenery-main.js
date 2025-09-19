// Copyright 2016-2025, University of Colorado Boulder

/**
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import '../../axon/js/main.js';
import '../../dot/js/main.js';
import '../../kite/js/main.js';
import '../../phet-core/js/main.js';
import '../../utterance-queue/js/main.js';
import './main.js';

if ( !window.hasOwnProperty( '_' ) ) {
  throw new Error( 'Underscore/Lodash not found: _' );
}

phet.scenery.Utils.polyfillRequestAnimationFrame();