// Copyright 2002-2014, University of Colorado

/**
 * A drawable that contains a group of rendered drawables, usually handled directly by a backbone.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
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
      this.used = true; // flag handled in the stitch
      
      // written in notifyInterval, should not be modified except with that.
      this.firstDrawable = null;
      this.lastDrawable = null;
      
      // linked-list handling for blocks
      this.previousBlock = null;
      this.nextBlock = null;
      
      // last set z-index, valid if > 0.
      this.zIndex = 0;
      
      if ( assertSlow ) {
        this.drawableList = cleanArray( this.drawableList );
      }
      
      return this;
    },
    
    dispose: function() {
      assert && assert( this.drawableCount === 0, 'There should be no drawables on a block when it is disposed' );
      
      // clear references
      this.display = null;
      this.firstDrawable = null;
      this.lastDrawable = null;
      
      this.previousBlock = null;
      this.nextBlock = null;
      
      if ( assertSlow ) {
        cleanArray( this.drawableList );
      }
      
      Drawable.prototype.dispose.call( this );
    },
    
    addDrawable: function( drawable ) {
      this.drawableCount++;
      this.markDirtyDrawable( drawable );
      
      if ( assertSlow ) {
        var idx = _.indexOf( this.drawableList, drawable );
        assertSlow && assertSlow( idx === -1, 'Drawable should not be added when it has not been removed' );
        this.drawableList.push( drawable );
        
        assertSlow && assertSlow( this.drawableCount === this.drawableList.length, 'Count sanity check, to make sure our assertions are not buggy' );
      }
    },
    
    removeDrawable: function( drawable ) {
      this.drawableCount--;
      
      if ( assertSlow ) {
        var idx = _.indexOf( this.drawableList, drawable );
        assertSlow && assertSlow( idx !== -1, 'Drawable should be already added when it is removed' );
        this.drawableList.splice( idx, 1 );
        
        assertSlow && assertSlow( this.drawableCount === this.drawableList.length, 'Count sanity check, to make sure our assertions are not buggy' );
      }
    },
    
    notifyInterval: function( firstDrawable, lastDrawable ) {
      this.firstDrawable = firstDrawable;
      this.lastDrawable = lastDrawable;
    },
    
    audit: function( allowPendingBlock, allowPendingList, allowDirty ) {
      if ( assertSlow ) {
        Drawable.prototype.audit.call( this, allowPendingBlock, allowPendingList, allowDirty );
        
        var count = 0;
        
        if ( !allowPendingList ) {
          
          // audit children, and get a count
          for ( var drawable = this.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
            drawable.audit( allowPendingBlock, allowPendingList, allowDirty );
            count++;
            if ( drawable === this.lastDrawable ) { break; }
          }
          
          if ( !allowPendingBlock ) {
            assertSlow && assertSlow( count === this.drawableCount, 'drawableCount should match' );
            
            // scan through to make sure our drawable lists are identical
            for ( var d = this.firstDrawable; d !== null; d = d.nextDrawable ) {
              assertSlow && assertSlow( d.parentDrawable === this, 'This block should be this drawable\'s parent' );
              assertSlow && assertSlow( _.indexOf( this.drawableList, d ) >= 0 );
              if ( d === this.lastDrawable ) { break; }
            }
          }
        }
      }
    }
  } );
  
  return Block;
} );
