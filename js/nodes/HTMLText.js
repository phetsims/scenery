// Copyright 2013-2015, University of Colorado Boulder

/**
 * HTML Text, with the same interface as Text
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Text = require( 'SCENERY/nodes/Text' ); // inherits from Text

  /**
   * NOTE: Currently does not properly handle multi-line (<br>) text height, since it expects DOM text that will be an
   * inline element
   */
  scenery.HTMLText = function HTMLText( text, options ) {
    // internal flag for Text
    this._isHTML = true;

    Text.call( this, text, options );
  };
  var HTMLText = scenery.HTMLText;

  inherit( Text, HTMLText, {} );

  return HTMLText;
} );


