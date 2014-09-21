// Copyright 2002-2013, University of Colorado

/**
 * Configuration file for development purposes, NOT for production deployments.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

// if has.js is included, set assertion flags to true (so we can catch errors during development)
if ( window.has ) {
  // default config only enables basic assertions
  window.has.add( 'assert.basic', function( global, document, anElement ) {
    'use strict';
    return true;
  } );
  // window.has.add( 'assert.slow', function( global, document, anElement ) { 'use strict'; return true; } );
}

// flag is set so we can ensure that the config has executed. This prevents various Require.js dynamic loading timeouts and script errors
window.loadedSceneryConfig = true;

require.config( {
  // depends on all of Scenery, Kite, and Dot
  deps: [ 'main', 'KITE/main', 'DOT/main', 'PHET_CORE/main' ],

  paths: {
    underscore: '../../sherpa/lodash-2.4.1',
    jquery: '../../sherpa/jquery-2.1.0',
    SCENERY: '.',
    KITE: '../../kite/js',
    DOT: '../../dot/js',
    PHET_CORE: '../../phet-core/js',
    ASSERT: '../../assert/js',
    AXON: '../../axon/js'
  },

  shim: {
    underscore: { exports: '_' },
    jquery: { exports: '$' }
  },

  urlArgs: new Date().getTime() // add cache buster query string to make browser refresh actually reload everything
} );
