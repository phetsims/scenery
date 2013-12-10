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
  
  scenery.DOMElementDrawable = function DOMElementDrawable( trail, renderer, domElement ) {
    Drawable.call( this, trail, renderer );
    this.domElement = domElement;
    
    this.dirty = true;
  };
  var DOMElementDrawable = scenery.DOMElementDrawable;
  
  inherit( Drawable, DOMElementDrawable, {
    // called from the Node that we called attachDOMDrawable on. should never be called after detachDOMDrawable.
    markDirty: function() {
      if ( !this.dirty ) {
        this.dirty = true;
        
        // TODO: notify what we want to call update() later
        if ( this.block ) {
          this.block.markDOMDirty( this );
        }
      }
    },
    
    update: function() {
    },
    
    dispose: function() {
      // super call
      Drawable.prototype.dispose.call( this );
    }
  } );
  
  return DOMElementDrawable;
} );
