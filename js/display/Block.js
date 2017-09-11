// Copyright 2014-2016, University of Colorado Boulder

/**
 * A specialized drawable for a layer of drawables with the same renderer (basically, it's a Canvas element, SVG
 * element, or some type of DOM container). Doesn't strictly have to have its DOM element used directly (Canvas block
 * used for caches).  This type is abstract, and meant to be subclassed.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  function Block( display, renderer ) {
    throw new Error( 'Should never be called' );
  }

  scenery.register( 'Block', Block );

  inherit( Drawable, Block, {

    /**
     * @param {Display} display
     * @param {number} renderer
     * @returns {Block} - Returns 'this' reference, for chaining
     */
    initializeBlock: function( display, renderer ) {
      this.initializeDrawable( renderer );
      this.display = display;
      this.drawableCount = 0;
      this.used = true; // flag handled in the stitch

      this.firstDrawable = null;
      this.lastDrawable = null;
      this.pendingFirstDrawable = null;
      this.pendingLastDrawable = null;

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
      this.pendingFirstDrawable = null;
      this.pendingLastDrawable = null;

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
      this.markDirty();

      if ( assertSlow ) {
        var idx = _.indexOf( this.drawableList, drawable );
        assertSlow && assertSlow( idx !== -1, 'Drawable should be already added when it is removed' );
        this.drawableList.splice( idx, 1 );

        assertSlow && assertSlow( this.drawableCount === this.drawableList.length, 'Count sanity check, to make sure our assertions are not buggy' );
      }
    },

    // @protected
    onIntervalChange: function( firstDrawable, lastDrawable ) {
      // stub, should be filled in with behavior in blocks
    },

    updateInterval: function() {
      if ( this.pendingFirstDrawable !== this.firstDrawable ||
           this.pendingLastDrawable !== this.lastDrawable ) {
        this.onIntervalChange( this.pendingFirstDrawable, this.pendingLastDrawable );

        this.firstDrawable = this.pendingFirstDrawable;
        this.lastDrawable = this.pendingLastDrawable;
      }
    },

    notifyInterval: function( firstDrawable, lastDrawable ) {
      this.pendingFirstDrawable = firstDrawable;
      this.pendingLastDrawable = lastDrawable;

      this.updateInterval();
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

            assertSlow && assertSlow( this.firstDrawable === this.pendingFirstDrawable, 'No pending first drawable' );
            assertSlow && assertSlow( this.lastDrawable === this.pendingLastDrawable, 'No pending last drawable' );

            // scan through to make sure our drawable lists are identical
            for ( var d = this.firstDrawable; d !== null; d = d.nextDrawable ) {
              assertSlow && assertSlow( d.renderer === this.renderer, 'Renderers should match' );
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
