// Copyright 2002-2012, University of Colorado

/**
 * Configuration file for production deployment purposes, NOT for development (it currently excludes most assertions).
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

// if has.js is included, set assertion flags to false, for running speed.
if ( window.has ) {
  window.has.add( 'assert.dot', function( global, document, anElement ) {
    return false;
  } );
  window.has.add( 'assert.kite', function( global, document, anElement ) {
    return false;
  } );
  window.has.add( 'assert.kite.extra', function( global, document, anElement ) {
    return false;
  } );
  window.has.add( 'assert.scenery', function( global, document, anElement ) {
    return false;
  } );
  window.has.add( 'assert.scenery.extra', function( global, document, anElement ) {
    return false;
  } );
}

require.config( {
  // depends on all of Scenery, Kite, and Dot
  deps: [ 'main', 'KITE/main', 'DOT/main', 'PHET_CORE/main' ],
  
  paths: {
    underscore: '../contrib/lodash.min-1.0.0-rc.3',
    jquery: '../contrib/jquery-1.8.3.min',
    SCENERY: '.',
    KITE: '../common/kite/js',
    DOT: '../common/dot/js',
    PHET_CORE: '../common/phet-core/js',
    ASSERT: '../common/assert/js'
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
