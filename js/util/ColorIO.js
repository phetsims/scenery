// Copyright 2016, University of Colorado Boulder

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
  var phetioInherit = require( 'TANDEM/phetioInherit' );
  var scenery = require( 'SCENERY/scenery' );

  // ifphetio
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );

  /**
   * IO type for phet/scenery's Color class.
   * @param {Color} color
   * @param phetioID
   * @constructor
   */
  function ColorIO( color, phetioID ) {
    assert && assertInstanceOf( color, Color );
    ObjectIO.call( this, color, phetioID );
  }

  phetioInherit( ObjectIO, 'ColorIO', ColorIO, {}, {
    documentation: 'A color, with rgba',

    /**
     * Encodes a Color into a state object.
     * @param {Color} color
     * @returns {Object}
     * @override
     */
    toStateObject: function( color ) {
      assert && assertInstanceOf( color, Color );
      return color.toStateObject();
    },

    /**
     * Decodes a state into a Color.
     * Use stateObject as the Font constructor's options argument
     * @param {Object} stateObject
     * @returns {Color}
     * @override
     */
    fromStateObject: function( stateObject ) {
      return new Color( stateObject.r, stateObject.g, stateObject.b, stateObject.a );
    }
  } );

  scenery.register( 'ColorIO', ColorIO );

  return ColorIO;
} );

