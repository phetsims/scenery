// Copyright 2002-2013, University of Colorado

/**
 * A unit that is drawable with a specific renderer
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Drawable = function Drawable( renderer ) {
    // what drawble we are being rendered (or put) into (will be filled in later)
    this.parentDrawable = null;
    
    // linked list handling (will be filled in later)
    this.previousDrawable = null;
    this.nextDrawable = null;
    
    this.renderer = renderer;
  };
  var Drawable = scenery.Drawable;
  
  inherit( Object, Drawable, {
    
    markDirty: function() {
      if ( !this.dirty ) {
        this.dirty = true;
        
        // TODO: notify what we want to call repaint() later
        if ( this.parentDrawable ) {
          this.parentDrawable.markDirtyInstance( this );
        }
      }
    },
    
    dispose: function() {
      
    },
    
    /*---------------------------------------------------------------------------*
    * Canvas API
    *----------------------------------------------------------------------------*/
    
    paintCanvas: function( wrapper, offset ) {
      
    }
    
    /*---------------------------------------------------------------------------*
    * SVG API
    *----------------------------------------------------------------------------*/
    
    
    
  } );
  
  return Drawable;
} );
