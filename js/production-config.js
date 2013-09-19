// Copyright 2002-2013, University of Colorado

/**
 * Configuration file for production deployment purposes, NOT for development (it currently excludes most assertions).
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

// if has.js is included, set assertion flags to false, for running speed.
if ( window.has ) {
  window.has.add( 'assert.dot', function( global, document, anElement ) {
    'use strict';
    return false;
  } );
  window.has.add( 'assert.kite', function( global, document, anElement ) {
    'use strict';
    return false;
  } );
  window.has.add( 'assert.kite.extra', function( global, document, anElement ) {
    'use strict';
    return false;
  } );
  window.has.add( 'assert.scenery', function( global, document, anElement ) {
    'use strict';
    return false;
  } );
  window.has.add( 'assert.scenery.extra', function( global, document, anElement ) {
    'use strict';
    return false;
  } );
}

require.config( {
  // depends on all of Scenery, Kite, and Dot
  deps: [ 'main', 'KITE/main', 'DOT/main', 'PHET_CORE/main' ],
  
  paths: {
    underscore: '../../sherpa/lodash-2.0.0',
    jquery: '../../sherpa/jquery-2.0.3',
    SCENERY: '.',
    KITE: '../../kite/js',
    DOT: '../../dot/js',
    PHET_CORE: '../../phet-core/js',
    ASSERT: '../../assert/js'
  },
  
  shim: {
    underscore: {
      exports: '_'
    },
    jquery: {
      exports: '$'
    }
  },
  
  urlArgs: new Date().getTime() // add cache buster query string to make browser refresh actually reload everything
} );
