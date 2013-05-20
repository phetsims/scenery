// Copyright 2002-2013, University of Colorado

/**
 * Configuration file for development purposes, NOT for production deployments.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

// if has.js is included, set assertion flags to true (so we can catch errors during development)
if ( window.has ) {
  window.has.add( 'assert.dot', function( global, document, anElement ) {
    'use strict';
    return true;
  } );
  window.has.add( 'assert.kite', function( global, document, anElement ) {
    'use strict';
    return true;
  } );
  window.has.add( 'assert.kite.extra', function( global, document, anElement ) {
    'use strict';
    return true;
  } );
  window.has.add( 'assert.scenery', function( global, document, anElement ) {
    'use strict';
    return true;
  } );
  window.has.add( 'assert.scenery.extra', function( global, document, anElement ) {
    'use strict';
    return true;
  } );
}

// flag is set so we can ensure that the config has executed. This prevents various Require.js dynamic loading timeouts and script errors
window.loadedSceneryConfig = true;

require.config( {
  // depends on all of Scenery, Kite, and Dot
  deps: [ 'main', 'KITE/main', 'DOT/main', 'PHET_CORE/main' ],
  
  paths: {
    underscore: '../contrib/lodash.min-1.0.0-rc.3',
    jquery: '../contrib/jquery-2.0.0.min',
    SCENERY: '.',
    KITE: '../common/kite/js',
    DOT: '../common/dot/js',
    PHET_CORE: '../common/phet-core/js',
    ASSERT: '../common/assert/js'
  },
  
  shim: {
    underscore: { exports: '_' },
    jquery: { exports: '$' }
  },
  
  urlArgs: new Date().getTime() // add cache buster query string to make browser refresh actually reload everything
} );
