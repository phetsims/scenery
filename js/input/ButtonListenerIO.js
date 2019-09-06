// Copyright 2017-2019, University of Colorado Boulder

/**
 * IO type for SCENERY ButtonListener (not SUN ButtonListener)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var ObjectIO = require( 'TANDEM/types/ObjectIO' );
  var scenery = require( 'SCENERY/scenery' );

  class ButtonListenerIO extends ObjectIO {}

  ButtonListenerIO.documentation = 'Button listener';
  ButtonListenerIO.events = [ 'up', 'over', 'down', 'out', 'fire' ];
  ButtonListenerIO.validator = { valueType: scenery.ButtonListener };
  ButtonListenerIO.typeName = 'ButtonListenerIO';
  ObjectIO.validateSubtype( ButtonListenerIO );

  return scenery.register( 'ButtonListenerIO', ButtonListenerIO );
} );