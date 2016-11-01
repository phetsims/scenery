// Copyright 2016, University of Colorado Boulder

/**
 * Tandem type for the PhET Scenery Font class
 * @author Andrew Adare (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var assertInstanceOf = require( 'PHET_IO/assertions/assertInstanceOf' );
  var phetioInherit = require( 'PHET_IO/phetioInherit' );
  var phetioNamespace = require( 'PHET_IO/phetioNamespace' );
  var TObject = require( 'PHET_IO/types/TObject' );

  /**
   * @constructor
   *
   * @param {Font} font - An instance of a phet.scenery.Font type
   * @param {string} phetioID - Full name of this font instance
   */
  function TFont( font, phetioID ) {
    TObject.call( this, font, phetioID );
    assertInstanceOf( font, phet.scenery.Font );
  }

  phetioInherit( TObject, 'TFont', TFont, {}, {
    documentation: 'Font handling for text drawing. Options:',

    // Serialize this font's configuration to an options object
    toStateObject: function( font ) {
      return {
        style: font.getStyle(),
        variant: font.getVariant(),
        weight: font.getWeight(),
        stretch: font.getStretch(),
        size: font.getSize(),
        lineHeight: font.getLineHeight(),
        family: font.getFamily()
      };
    },

    // Use stateObject as the Font constructor's options argument
    fromStateObject: function( stateObject ) {
      return new phet.scenery.Font( stateObject );
    }
  } );

  phetioNamespace.register( 'TFont', TFont );

  return TFont;
} );
