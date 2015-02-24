// Copyright 2002-2014, University of Colorado Boulder

/**
 * Configuration file for development and production deployments.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

require.config( {
  // depends on all of Scenery, Kite, Dot, Axon and phet-core
  deps: [ 'main', 'KITE/main', 'DOT/main', 'AXON/main', 'PHET_CORE/main' ],

  paths: {

    // plugins
    image: '../../chipper/js/requirejs-plugins/image',
    text: '../../sherpa/text',

    SCENERY: '.',
    KITE: '../../kite/js',
    DOT: '../../dot/js',
    PHET_CORE: '../../phet-core/js',
    AXON: '../../axon/js'
  },

  // optional cache buster to make browser refresh load all included scripts, can be disabled with ?cacheBuster=false
  urlArgs: Date.now()
} );
