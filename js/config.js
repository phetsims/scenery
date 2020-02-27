
// Copyright 2013-2019, University of Colorado Boulder

/**
 * Configuration file for development and production deployments.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

require.config( {

// depends on all of Scenery, Kite, Dot, Axon and phet-core
  deps: [ 'main', '/kite/js/main', '/dot/js/main', '/axon/js/main', '/phet-core/js/main', '/utterance-queue/js/main' ],

  paths: {

    // plugins
    image: '../../chipper/js/requirejs-plugins/image',
    ifphetio: '../../chipper/js/requirejs-plugins/ifphetio',

    // third-party libs
    text: '../../sherpa/lib/text-2.0.12',

    AXON: '../../axon/js',

    DOT: '../../dot/js',
    KITE: '../../kite/js',
    PHET_CORE: '../../phet-core/js',
    SCENERY: '.',
    TANDEM: '../../tandem/js',
    UTTERANCE_QUEUE: '../../utterance-queue/js',

    REPOSITORY: '..'
  },

// optional cache bust to make browser refresh load all included scripts, can be disabled with ?cacheBust=false
  urlArgs: 'bust=' + ( new Date() ).getTime()
} );