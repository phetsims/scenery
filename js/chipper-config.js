// Copyright 2002-2014, University of Colorado Boulder

/**
 * Configuration file for development purposes, NOT for production deployments.
 * This configuation file adds some auxiliary phetsims repos such as chipper and energy-skate-park-basics.
 * Just for testing WebGL development.  Most uses should be using config.js instead.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

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
    AXON: '../../axon/js',
    ENERGY_SKATE_PARK_BASICS: '../../energy-skate-park-basics/js',
    text: '../../sherpa/text',

    image: '../../chipper/js/requirejs-plugins/image'
  },

  shim: {
    underscore: { exports: '_' },
    jquery: { exports: '$' }
  },

  // optional cache buster to make browser refresh load all included scripts, can be disabled with ?cacheBuster=false
  urlArgs: phet.chipper.getCacheBusterArgs()
} );
