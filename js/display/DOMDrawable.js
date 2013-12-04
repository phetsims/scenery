// Copyright 2002-2013, University of Colorado

/**
 * TODO docs
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.DOMDrawable = function DOMDrawable( drawableState, domElement ) {
    Drawable.call( this, drawableState );
    
    this._domElement = domElement;
    
    // TODO: handle transforms?
  };
  var DOMDrawable = scenery.DOMDrawable;
  
  inherit( Drawable, DOMDrawable, {
    getDomElement: function() {
      return this._domElement;
    }
  } );
  
  return DOMDrawable;
} );
