// Copyright 2002-2013, University of Colorado

/**
 * DOM drawable for a specific DOM element. TODO docs
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.DOMElementDrawable = function DOMElementDrawable( trail, renderer, domElement, repaintCallback ) {
    Drawable.call( this, trail, renderer );
    this.domElement = domElement;
    this.repaintCallback = repaintCallback;
    
    this.dirty = true;
  };
  var DOMElementDrawable = scenery.DOMElementDrawable;
  
  inherit( Drawable, DOMElementDrawable, {
    repaint: function() {
      this.repaintCallback && this.repaintCallback();
    },
    
    dispose: function() {
      // super call
      Drawable.prototype.dispose.call( this );
    }
  } );
  
  return DOMElementDrawable;
} );
