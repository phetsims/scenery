// Copyright 2016, University of Colorado Boulder

/**
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Andrew Adare
 */
define( function( require ) {
  'use strict';

  // modules
  var assertInstanceOf = require( 'PHET_IO/assertions/assertInstanceOf' );
  var phetioInherit = require( 'PHET_IO/phetioInherit' );
  var phetioNamespace = require( 'PHET_IO/phetioNamespace' );
  var TObject = require( 'PHET_IO/types/TObject' );

  var TColor = function( color, phetioID ) {
    TObject.call( this, color, phetioID );
    assertInstanceOf( color, phet.scenery.Color );
  };

  phetioInherit( TObject, 'TColor', TColor, {}, {
    documentation: 'A color, with rgba',

    toStateObject: function( color ) {
      return color.toStateObject();
    },

    fromStateObject: function( stateObject ) {
      return new phet.scenery.Color( stateObject.r, stateObject.g, stateObject.b, stateObject.a );
    }
  } );

  phetioNamespace.register( 'TColor', TColor );

  return TColor;
} );

