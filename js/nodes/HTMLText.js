// Copyright 2002-2013, University of Colorado

/**
 * HTML Text, with the same interface as Text
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Text = require( 'SCENERY/nodes/Text' ); // inherits from Text

  scenery.HTMLText = function HTMLText( text, options ) {
    // internal flag for Text
    this._isHTML = true;

    Text.call( this, text, options );
  };
  var HTMLText = scenery.HTMLText;

  inherit( Text, HTMLText, {} );

  return HTMLText;
} );


