// Copyright 2002-2014, University of Colorado

/**
 * Stitcher that only rebuilds the parts necessary, and attempts greedy block matching as an optimization.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var scenery = require( 'SCENERY/scenery' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var Stitcher = require( 'SCENERY/display/Stitcher' );
  
  var prototype = {
    stitch: function( backbone, firstDrawable, lastDrawable, oldFirstDrawable, oldLastDrawable, firstChangeInterval, lastChangeInterval ) {
      this.initialize( backbone, firstDrawable, lastDrawable, oldFirstDrawable, oldLastDrawable, firstChangeInterval, lastChangeInterval );
      this.blockOrderChanged = false;
      this.reusableBlocks = cleanArray( this.reusableBlocks ); //re-use if possible
      
      var interval;
      
      // record current first/last drawables for the entire backbone
      this.recordBackboneBoundaries();
      
      // per-interval work
      for ( interval = firstChangeInterval; interval !== null; interval = interval.nextChangeInterval ) {
        assert && assert( !interval.isEmpty(), 'We now guarantee that the intervals are non-empty' );
        
        if ( backbone.blocks.length ) {
          //OHTWO TODO: here (in the old-iteration), we should collect references to potentially reusable blocks?
          this.noteIntervalForRemoval( backbone.display, interval, oldFirstDrawable, oldLastDrawable );
          
          var firstBlock = interval.drawableBefore === null ? backbone.blocks[0] : interval.drawableBefore.pendingParentDrawable;
          var lastBlock = interval.drawableAfter === null ? backbone.blocks[backbone.blocks.length-1] : interval.drawableAfter.pendingParentDrawable;
          
          // blocks totally contained within the change interval are marked as reusable (doesn't include end blocks)
          if ( firstBlock !== lastBlock ) {
            for ( var markedBlock = firstBlock.nextBlock; markedBlock !== lastBlock; markedBlock = markedBlock.nextBlock ) {
              markedBlock.used = false; // mark it as unused until we pull it out (so we can reuse, or quickly identify)
              this.reusableBlocks.push( markedBlock );
              this.removeBlock( markedBlock ); // remove it from our blocks array
            }
          }
        }
      }
      
      for ( interval = firstChangeInterval; interval !== null; interval = interval.nextChangeInterval ) {
        /*---------------------------------------------------------------------------*
        * Interval start
        *----------------------------------------------------------------------------*/
        
        // For each virtual block, once set, the drawables will be added to this block. At the start of an interval
        // if there is a block tied to the drawableBefore, we will use it. Otherwise, as we go through the drawables,
        // we attempt to match previously-used "reusable" blocks. 
        var currentBlock = interval.drawableBefore ?
                           interval.drawableBefore.pendingParentDrawable :
                           null;
        var previousBlock = null;
        
        // The first drawable that will be in the "range of drawables to be added to the block". This excludes the
        // "unchanged endpoint" drawables, and only includes "internal" drawables.
        var firstDrawableForBlockChange = null;
        
        var boundaryCount = 0;
        
        var previousDrawable = interval.drawableBefore; // possibly null
        var drawable = interval.drawableBefore ? interval.drawableBefore.nextDrawable : firstDrawable;
        for ( ; drawable !== lastDrawable; drawable = drawable.nextDrawable ) {
          if ( previousDrawable && this.hasGapBetweenDrawables( previousDrawable, drawable ) ) {
            /*---------------------------------------------------------------------------*
            * Interval boundary
            *----------------------------------------------------------------------------*/
            
            // get our final block reference, and add drawables to it
            currentBlock = this.addInternalDrawables( backbone, currentBlock, firstDrawableForBlockChange, previousDrawable );
            
            // link our blocks
            if ( boundaryCount > 0 ) {
              assert && assert( previousBlock, 'Should always have a previous block if this is not the first boundary' );
              assert && assert( firstDrawableForBlockChange && firstDrawableForBlockChange.previousDrawable,
                                'Should always have drawables surrounding the boundary' );
              this.linkBlocks( previousBlock, currentBlock, firstDrawableForBlockChange.previousDrawable, firstDrawableForBlockChange );
            } else if ( !interval.drawableBefore && firstDrawableForBlockChange ) {
              // we are the first block of our backbone at the start of an interval
              this.linkBlocks( null, currentBlock, null, firstDrawableForBlockChange );
            } else {
              // we are continuing in the middle of a block
            }
            
            previousBlock = currentBlock;
            currentBlock = null; // so we can match another
            
            boundaryCount++;
          }
          
          if ( drawable === interval.drawableAfter ) {
            // NOTE: leaves previousDrawable untouched, we will use it below
            break;
          } else {
            /*---------------------------------------------------------------------------*
            * Internal drawable
            *----------------------------------------------------------------------------*/
            
            // attempt to match for our block to use
            if ( currentBlock === null && drawable.parentDrawable && !drawable.parentDrawable.used ) {
              // mark our currentBlock to be used, but don't useBlock() it yet (we may end up gluing it at the
              // end of our interval).
              currentBlock = drawable.parentDrawable;
            }
            
            if ( firstDrawableForBlockChange === null ) {
              firstDrawableForBlockChange = drawable;
            }
          }
          
          // on to the next drawable
          previousDrawable = drawable;
        }
        
        /*---------------------------------------------------------------------------*
        * Interval end
        *----------------------------------------------------------------------------*/
        if ( boundaryCount === 0 && interval.drawableBefore && interval.drawableAfter &&
             interval.drawableBefore.pendingParentDrawable !== interval.drawableAfter.pendingParentDrawable ) {
          /*---------------------------------------------------------------------------*
          * Glue
          *----------------------------------------------------------------------------*/
          
          //OHTWO TODO: dynamically decide which end is better to glue on
          var oldNextBlock = interval.drawableAfter.pendingParentDrawable;
          
          // (optional?) mark the old block as reusable
          oldNextBlock.used = false;
          this.reusableBlocks.push( oldNextBlock );
          
          assert && assert( currentBlock && currentBlock === interval.drawableBefore.pendingParentDrawable );
          
          currentBlock = this.addInternalDrawables( backbone, currentBlock, firstDrawableForBlockChange, previousDrawable );
          this.moveExternalDrawables( backbone, interval, currentBlock, lastDrawable );
        } else if ( boundaryCount > 0 && interval.drawableBefore && interval.drawableAfter &&
                    interval.drawableBefore.pendingParentDrawable === interval.drawableAfter.pendingParentDrawable ) {
          //OHTWO TODO: with gluing, how do we handle the if statement block?
          /*---------------------------------------------------------------------------*
          * Unglue
          *----------------------------------------------------------------------------*/
          
          var firstUngluedDrawable = firstDrawableForBlockChange ? firstDrawableForBlockChange : interval.drawableAfter;
          currentBlock = this.ensureUsedBlock( currentBlock, backbone, firstUngluedDrawable );
          backbone.markDirtyDrawable( currentBlock );
          
          // internal additions
          if ( firstDrawableForBlockChange ) {
            this.notePendingAdditions( backbone, currentBlock, firstUngluedDrawable, previousDrawable );
          }
          this.moveExternalDrawables( backbone, interval, currentBlock, lastDrawable );
        } else {
          // handle a normal end-point, where we add our drawables to our last block
          
          // use the "after" block, if it is available
          if ( interval.drawableAfter ) {
            assert && assert( interval.drawableAfter.pendingParentDrawable );
            currentBlock = interval.drawableAfter.pendingParentDrawable;
          }
          currentBlock = this.addInternalDrawables( backbone, currentBlock, firstDrawableForBlockChange, previousDrawable );
          
          // link our blocks
          if ( boundaryCount > 0 ) {
            assert && assert( previousBlock, 'Should always have a previous block if this is not the first boundary' );
            assert && assert( firstDrawableForBlockChange && firstDrawableForBlockChange.previousDrawable,
                                'Should always have drawables surrounding the boundary' );
            this.linkBlocks( previousBlock, currentBlock, firstDrawableForBlockChange.previousDrawable, firstDrawableForBlockChange );
          } else if ( !interval.drawableBefore && firstDrawableForBlockChange ) {
            // we are the first block of our backbone at the start of an interval
            this.linkBlocks( null, currentBlock, null, firstDrawableForBlockChange );
          } else {
            // we are continuing in the middle of a block
          }
        }
      }
      //OHTWO TODO: maintain array or linked-list of blocks (and update)
      //OHTWO TODO: remember to set blockOrderChanged on changes  (everything removed?)
      //OHTWO VERIFY: DOMBlock special case with backbones / etc.? Always have the same drawable!!!
      
      this.removeUnusedBlocks( backbone );
      
      //OHTWO TODO: just use direct operations!
      if ( this.blockOrderChanged ) {
        this.processBlockLinkedList( backbone, firstDrawable.pendingParentDrawable, lastDrawable.pendingParentDrawable );
        this.reindex();
      }
      
      this.clean();
      cleanArray( this.reusableBlocks );
    },
    
    noteIntervalForRemoval: function( display, interval, oldFirstDrawable, oldLastDrawable ) {
      // if before/after is null, we go out to the old first/last
      var first = interval.drawableBefore || oldFirstDrawable;
      var last = interval.drawableAfter || oldLastDrawable;
      
      for ( var drawable = first;; drawable = drawable.oldNextDrawable ) {
        this.notePendingRemoval( drawable );
        
        if ( drawable === last ) { break; }
      }
    },
    
    hasGapBetweenDrawables: function( a, b ) {
      return a.renderer !== b.renderer || Renderer.isDOM( a.renderer ) || Renderer.isDOM( b.renderer );
    },
    
    addInternalDrawables: function( backbone, currentBlock, firstDrawableForBlockChange, lastDrawableForBlockChange ) {
      if ( firstDrawableForBlockChange ) {
        currentBlock = this.ensureUsedBlock( currentBlock, backbone, firstDrawableForBlockChange );
        
        this.notePendingAdditions( backbone, currentBlock, firstDrawableForBlockChange, lastDrawableForBlockChange );
      }
      return currentBlock;
    },
    
    moveExternalDrawables: function( backbone, interval, block, lastStitchDrawable ) {
      var firstDrawable = interval.drawableAfter;
      if ( firstDrawable ) {
        var lastDrawable = lastStitchDrawable;
        while ( interval.nextChangeInterval ) {
          interval = interval.nextChangeInterval;
          if ( !interval.isEmpty() ) {
            lastDrawable = interval.drawableBefore;
            break;
          }
        }
        
        this.notePendingMoves( backbone, block, firstDrawable, lastDrawable );
      }
    },
    
    notePendingAdditions: function( backbone, block, firstDrawable, lastDrawable ) {
      for ( var drawable = firstDrawable;; drawable = drawable.nextDrawable ) {
        this.notePendingAddition( drawable, block );
        if ( drawable === lastDrawable ) { break; }
      }
    },
    
    notePendingMoves: function( backbone, block, firstDrawable, lastDrawable ) {
      for ( var drawable = firstDrawable;; drawable = drawable.nextDrawable ) {
        assert && assert( !drawable.pendingAddition && !drawable.pendingRemoval,
                          'Moved drawables should be thought of as unchanged, and thus have nothing pending yet' );
        
        this.notePendingMove( drawable, block );
        if ( drawable === lastDrawable ) { break; }
      }
    },
    
    // If there is no currentBlock, we create one to match. Otherwise if the currentBlock is marked as 'unused' (i.e.
    // it is in the reusableBlocks array), we mark it as used so it won't me matched elsewhere.
    ensureUsedBlock: function( currentBlock, backbone, someIncludedDrawable ) {
      // if we have a matched block (or started with one)
      if ( currentBlock ) {
        // since our currentBlock may be from reusableBlocks, we will need to mark it as used now.
        if ( !currentBlock.used ) {
          this.useBlock( backbone, currentBlock );
        }
      } else {
        // need to create one
        currentBlock = this.getBlockForRenderer( backbone, someIncludedDrawable.renderer, someIncludedDrawable );
      }
      return currentBlock;
    },
    
    // NOTE: this doesn't handle hooking up the block linked list
    getBlockForRenderer: function( backbone, renderer, drawable ) {
      var block;
      
      // If it's not a DOM block, scan our reusable blocks for a match
      if ( !Renderer.isDOM( renderer ) ) {
        // backwards scan, hopefully it's faster?
        for ( var i = this.reusableBlocks.length - 1; i >= 0; i-- ) {
          block = this.reusableBlocks[i];
          if ( block.renderer === renderer ) {
            this.reusableBlocks.splice( i, 1 ); // remove it from our reusable blocks, since it's now in use
            block.used = true; // mark it as used, so we don't match it when scanning
            break;
          }
        }
      }
      
      if ( !block ) {
        // Didn't find it in our reusable blocks, create a fresh one from scratch
        block = this.createBlock( renderer, drawable );
      }
      
      this.blockOrderChanged = true; // we created a new block, this will always happen
      
      return block;
    },
    
    // removes a block from the list of reused blocks (done during matching)
    useBlock: function( backbone, block ) {
      var idx = _.indexOf( this.reusableBlocks, block );
      assert && assert( idx >= 0 );
      
      // remove it
      this.reusableBlocks.splice( idx, 1 );
      
      // mark it as used
      block.used = true;
    },
    
    // removes them from our domElement, and marks them for disposal
    removeUnusedBlocks: function( backbone ) {
      while ( this.reusableBlocks.length ) {
        this.markBlockForDisposal( this.reusableBlocks.pop() );
      }
    },
    
    linkBlocks: function( beforeBlock, afterBlock, beforeDrawable, afterDrawable ) {
      assert && assert( ( beforeBlock === null && beforeDrawable === null ) ||
                        ( beforeBlock instanceof scenery.Block && beforeDrawable instanceof scenery.Drawable ) );
      assert && assert( ( afterBlock === null && afterDrawable === null ) ||
                        ( afterBlock instanceof scenery.Block && afterDrawable instanceof scenery.Drawable ) );
      
      if ( beforeBlock ) {
        if ( beforeBlock.nextBlock !== afterBlock ) {
          this.blockOrderChanged = true;
          
          // disconnect from the previously-connected block (if any)
          if ( beforeBlock.nextBlock ) {
            beforeBlock.nextBlock.previousBlock = null;
          }
          
          beforeBlock.nextBlock = afterBlock;
        }
        beforeBlock.pendingLastDrawable = beforeDrawable;
      }
      if ( afterBlock ) {
        if ( afterBlock.previousBlock !== beforeBlock ) {
          this.blockOrderChanged = true;
          
          // disconnect from the previously-connected block (if any)
          if ( afterBlock.previousBlock ) {
            afterBlock.previousBlock.nextBlock = null;
          }
          
          afterBlock.previousBlock = beforeBlock;
        }
        afterBlock.pendingFirstDrawable = afterDrawable;
      }
    },
    
    processBlockLinkedList: function( backbone, firstBlock, lastBlock ) {
      // for now, just clear out the array first
      while ( backbone.blocks.length ) {
        backbone.blocks.pop();
      }
      
      // and rewrite it
      for ( var block = firstBlock;; block = block.nextBlock ) {
        backbone.blocks.push( block );
        
        // if its first/last drawable has changed, update it
        block.updateInterval();
        
        if ( block === lastBlock ) { break; }
      }
    }
  };
  
  var GreedyStitcher = scenery.GreedyStitcher = inherit( Stitcher, function GreedyStitcher() {
    // nothing done
  }, prototype );
  
  GreedyStitcher.stitchPrototype = prototype;
  
  return GreedyStitcher;
} );
