// Copyright 2017-2019, University of Colorado Boulder

/**
 * IO type for Color
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var Color = require( 'SCENERY/util/Color' );
  var ObjectIO = require( 'TANDEM/types/ObjectIO' );
  var scenery = require( 'SCENERY/scenery' );
  var validate = require( 'AXON/validate' );

  class ColorIO extends ObjectIO {

    /**
     * Encodes a Color into a state object.
     * @param {Color} color
     * @returns {Object}
     * @override
     */
    static toStateObject( color ) {
      validate( color, this.validator );
      return color.toStateObject();
    }

    /**
     * Decodes a state into a Color.
     * Use stateObject as the Font constructor's options argument
     * @param {Object} stateObject
     * @returns {Color}
     * @override
     */
    static fromStateObject( stateObject ) {
      return new Color( stateObject.r, stateObject.g, stateObject.b, stateObject.a );
    }
  }

  ColorIO.documentation = 'A color, with rgba';
  ColorIO.validator = { valueType: Color };
  ColorIO.typeName = 'ColorIO';
  ObjectIO.validateSubtype( ColorIO );

  return scenery.register( 'ColorIO', ColorIO );
} );

