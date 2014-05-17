// Copyright 2002-2014, University of Colorado

/**
 * A drawable that contains a group of rendered drawables, usually handled directly by a backbone.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.Block = function Block( display, renderer ) {
    throw new Error( 'Should never be called' );
  };
  var Block = scenery.Block;
  
  inherit( Drawable, Block, {
    initializeBlock: function( display, renderer ) {
      this.initializeDrawable( renderer );
      this.display = display;
      this.drawableCount = 0;
      
      this.firstDrawable = null;
      this.lastDrawable = null;
      
      return this;
    },
    
    dispose: function() {
      assert && assert( this.drawableCount === 0, 'There should be no drawables on a block when it is disposed' );
      
      // clear references
      this.display = null;
      this.firstDrawable = null;
      this.lastDrawable = null;
      
      Drawable.prototype.dispose.call( this );
    },
    
    addDrawable: function( drawable ) {
      drawable.parentDrawable = this;
      this.drawableCount++;
      this.markDirtyDrawable( drawable );
    },
    
    removeDrawable: function( drawable ) {
      this.drawableCount--;
      drawable.parentDrawable = null;
    },
    
    notifyInterval: function( firstDrawable, lastDrawable ) {
      this.firstDrawable = firstDrawable;
      this.lastDrawable = lastDrawable;
    }
  } );
  
  return Block;
} );
