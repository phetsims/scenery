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
  
  function hasGapBetweenDrawables( a, b ) {
    return a.renderer !== b.renderer || Renderer.isDOM( a.renderer ) || Renderer.isDOM( b.renderer );
  }
  
  function isOpenBefore( drawable ) {
    return drawable.previousDrawable !== null && !hasGapBetweenDrawables( drawable.previousDrawable, drawable );
  }
  
  function isOpenAfter( drawable ) {
    return drawable.nextDrawable !== null && !hasGapBetweenDrawables( drawable, drawable.nextDrawable );
  }
  
  function intervalHasNewInternals( interval, firstStitchDrawable, lastStitchDrawable ) {
    if ( interval.drawableBefore ) {
      return interval.drawableBefore.nextDrawable !== interval.drawableAfter; // OK for after to be null
    }
    else if ( interval.drawableAfter ) {
      return interval.drawableAfter.previousDrawable !== interval.drawableBefore; // OK for before to be null
    }
    else {
      return firstStitchDrawable !== null;
    }
  }
  
  function intervalHasOldInternals( interval, oldFirstStitchDrawable, oldLastStitchDrawable ) {
    if ( interval.drawableBefore ) {
      return interval.drawableBefore.oldNextDrawable !== interval.drawableAfter; // OK for after to be null
    }
    else if ( interval.drawableAfter ) {
      return interval.drawableAfter.oldPreviousDrawable !== interval.drawableBefore; // OK for before to be null
    }
    else {
      return oldFirstStitchDrawable !== null;
    }
  }
  
  var prototype = {
    stitch: function( backbone, firstStitchDrawable, lastStitchDrawable, oldFirstStitchDrawable, oldLastStitchDrawable, firstChangeInterval, lastChangeInterval ) {
      this.initialize( backbone, firstStitchDrawable, lastStitchDrawable, oldFirstStitchDrawable, oldLastStitchDrawable, firstChangeInterval, lastChangeInterval );
      
      // Tracks whether our order of blocks changed. If it did, we'll need to rebuild our blocks array. This flag is
      // set if we remove any blocks, create any blocks, or change the order between two blocks (via linkBlocks).
      // It does NOT occur in unuseBlock, since we may reuse the same block in the same position (thus not having an
      // order change).
      this.blockOrderChanged = false;
      
      // List of blocks that (in the current part of the stitch being processed) are not set to be used by any
      // drawables. Blocks are added to here when they are fully internal to a change interval, and when we glue
      // blocks together. They can be reused through the block-matching process. If they are not reused at the end of
      // this stitch, they will be marked for removal.
      this.reusableBlocks = cleanArray( this.reusableBlocks ); // re-use instance, since we are effectively pooled
      
      var interval;
      
      // record current first/last drawables for the entire backbone
      this.recordBackboneBoundaries();
      
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( 'phase 1: old linked list' );
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();
      
      // handle pending removal of old blocks/drawables
      if ( backbone.blocks.length ) {
        for ( interval = firstChangeInterval; interval !== null; interval = interval.nextChangeInterval ) {
          assert && assert( !interval.isEmpty(), 'We now guarantee that the intervals are non-empty' );
        
          // note pending removal on all drawables in the old linked list for the interval.
          // this only makes sense on intervals that have old "internal" drawables
          if ( intervalHasOldInternals( interval, oldFirstStitchDrawable, oldLastStitchDrawable ) ) {
            var firstRemoval = interval.drawableBefore ? interval.drawableBefore.oldNextDrawable : oldFirstStitchDrawable;
            var lastRemoval = interval.drawableAfter ? interval.drawableAfter.oldPreviousDrawable : oldLastStitchDrawable;
            
            for ( var removedDrawable = firstRemoval;; removedDrawable = removedDrawable.oldNextDrawable ) {
              this.notePendingRemoval( removedDrawable );
              if ( removedDrawable === lastRemoval ) { break; }
            }
          }
          
          var firstBlock = interval.drawableBefore === null ? backbone.blocks[0] : interval.drawableBefore.pendingParentDrawable;
          var lastBlock = interval.drawableAfter === null ? backbone.blocks[backbone.blocks.length-1] : interval.drawableAfter.pendingParentDrawable;
          
          // blocks totally contained within the change interval are marked as reusable (doesn't include end blocks)
          if ( firstBlock !== lastBlock ) {
            for ( var markedBlock = firstBlock.nextBlock; markedBlock !== lastBlock; markedBlock = markedBlock.nextBlock ) {
              this.unuseBlock( markedBlock );
            }
          }
        }
      }
      
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();
      
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( 'phase 2: new linked list' );
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();
      
      // don't process the single interval left if we aren't left with any drawables (thus left with no blocks)
      if ( firstStitchDrawable ) {
        for ( interval = firstChangeInterval; interval !== null; interval = interval.nextChangeInterval ) {
          this.processInterval( interval, firstStitchDrawable, lastStitchDrawable );
        }
      }
      
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();
      
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( 'phase 3: cleanup' );
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();
      
      //OHTWO VERIFY: remember to set blockOrderChanged on changes  (everything removed?)
      
      this.removeUnusedBlocks( backbone );
      
      // since we use markBeforeBlock/markAfterBlock
      this.updateBlockIntervals();
      
      if ( firstStitchDrawable === null ) {
        this.useNoBlocks();
      } else if ( this.blockOrderChanged ) {
        this.processBlockLinkedList( backbone, firstStitchDrawable.pendingParentDrawable, lastStitchDrawable.pendingParentDrawable );
        this.reindex();
      }
      
      this.clean();
      cleanArray( this.reusableBlocks );
      
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();
    },
    
    processInterval: function( interval, firstStitchDrawable, lastStitchDrawable ) {
      assert && assert( interval instanceof scenery.ChangeInterval );
      assert && assert( firstStitchDrawable instanceof scenery.Drawable, 'We assume we have a non-null remaining section' );
      assert && assert( lastStitchDrawable instanceof scenery.Drawable, 'We assume we have a non-null remaining section' );
      assert && assert( !interval.isEmpty(), 'We now guarantee that the intervals are non-empty' );
      
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'interval: ' +
                                                                          ( interval.drawableBefore ? interval.drawableBefore.toString() : 'null' ) +
                                                                          ' to ' +
                                                                          ( interval.drawableAfter ? interval.drawableAfter.toString() : 'null' ) );
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();
      
      // we need to mark the start of the next interval (or the end), as it's the end of the next section of
      // external drawables, and we'll need to move externals up to it if gluing/ungluing
      var drawableBeforeNextInterval = interval.nextChangeInterval ? interval.nextChangeInterval.drawableBefore : lastStitchDrawable;
      
      // check if our interval removes everything, we may need a glue
      if ( !intervalHasNewInternals( interval, firstStitchDrawable, lastStitchDrawable ) ) {
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'no current internal drawables in interval' );
        
        // separate if, last condition above would cause issues with the normal operation branch
        if ( interval.drawableBefore && interval.drawableAfter ) {
          var beforeBlock = interval.drawableBefore.pendingParentDrawable;
          var afterBlock = interval.drawableAfter.pendingParentDrawable;
          
          // check if glue is needed
          if ( beforeBlock !== afterBlock ) {
            sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'glue (with non-internal)' );
            sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();
            
            // for now, toss the after block (simplifies changes in one direction)
            this.unuseBlock( afterBlock );
            this.notePendingMoves( beforeBlock, interval.drawableAfter, drawableBeforeNextInterval );
            if ( !drawableBeforeNextInterval.nextDrawable ) {
              this.linkAfterDrawable( drawableBeforeNextInterval );
            }
            
            sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
          }
        }
        
        if ( !isOpenAfter( interval.drawableBefore ) ) {
          this.linkAfterDrawable( interval.drawableBefore );
        }
      }
      // otherwise normal operation
      else {
        var drawable = interval.drawableBefore ? interval.drawableBefore.nextDrawable : firstStitchDrawable;
        
        // if we have any current drawables at all
        var subBlockFirstDrawable = null;
        var matchedBlock = null;
        var isFirst = true;
        
        while ( true ) {
          var nextDrawable = drawable.nextDrawable;
          var isLast = nextDrawable === interval.drawableAfter;
          
          assert && assert( nextDrawable !== null || isLast, 'If our nextDrawable is null, isLast must be true' );
          
          if ( !subBlockFirstDrawable ) {
            subBlockFirstDrawable = drawable;
          }
          
          if ( matchedBlock === null && drawable.parentDrawable && !drawable.parentDrawable.used ) {
            matchedBlock = drawable.parentDrawable;
          }
          
          if ( isLast || hasGapBetweenDrawables( drawable, nextDrawable ) ) {
            if ( isFirst ) {
              // we'll handle any glue/unglue at the start, so every processSubBlock can be set correctly.
              this.processEdgeCases( interval, subBlockFirstDrawable, drawable, isLast, drawableBeforeNextInterval );
            }
            this.processSubBlock( interval, subBlockFirstDrawable, drawable, matchedBlock, isFirst, isLast );
            subBlockFirstDrawable = null;
            matchedBlock = null;
            isFirst = false;
          }
          
          if ( isLast ) {
            break;
          } else {
            drawable = nextDrawable;
          }
        }
      }
      
      
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
    },
    
    // firstDrawable and lastDrawable refer to the specific sub-block
    processSubBlock: function( interval, firstDrawable, lastDrawable, matchedBlock, isFirst, isLast ) {
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'sub-block: ' + firstDrawable.toString() + ' to ' + lastDrawable.toString() );
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();
      
      var openBefore = isOpenBefore( firstDrawable );
      var openAfter = isOpenAfter( lastDrawable );
      
      assert && assert( !openBefore || isFirst, 'openBefore implies isFirst' );
      assert && assert( !openAfter || isLast, 'openAfter implies isLast' );
      
      assert && assert( !openBefore || !openAfter || firstDrawable.previousDrawable.pendingParentDrawable === lastDrawable.nextDrawable.pendingParentDrawable,
        'If we would use both the before and after blocks, make sure any gluing ' );
      
      // if our sub-block gets combined into the previous block, use its block instead of any match-scanned blocks
      if ( openBefore ) {
        matchedBlock = firstDrawable.previousDrawable.pendingParentDrawable;
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'combining into before block: ' + matchedBlock.toString() );
      }
      
      // if our sub-block gets combined into the next block, use its block instead of any match-scanned blocks
      if ( openAfter ) {
        matchedBlock = lastDrawable.nextDrawable.pendingParentDrawable;
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'combining into after block: ' + matchedBlock.toString() );
      }
      
      // create a block if matchedBlock is null, otherwise mark it as used (if it is in reusableBlocks)
      matchedBlock = this.ensureUsedBlock( matchedBlock, firstDrawable );
      
      // add internal drawables
      for ( var drawable = firstDrawable;; drawable = drawable.nextDrawable ) {
        this.notePendingAddition( drawable, matchedBlock );
        if ( drawable === lastDrawable ) { break; }
      }
      
      // link our blocks (and set pending block boundaries) as needed. assumes glue/unglue has already been performed
      if ( !openBefore ) {
        this.linkBeforeDrawable( firstDrawable );
      }
      if ( isLast && !openAfter ) {
        this.linkAfterDrawable( lastDrawable );
      }
      
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
    },
    
    // firstDrawable and lastDrawable refer to the specific sub-block (if it exists), isLast refers to if it's the last sub-block
    processEdgeCases: function( interval, firstDrawable, lastDrawable, isLast, drawableBeforeNextInterval ) {
      // this test passes for glue and unglue cases
      if ( firstDrawable.previousDrawable !== null && lastDrawable.nextDrawable !== null ) {
        var openBefore = isOpenBefore( firstDrawable );
        var openAfter = isOpenAfter( lastDrawable );
        var beforeBlock = firstDrawable.previousDrawable.pendingParentDrawable;
        var afterBlock = drawableBeforeNextInterval.pendingParentDrawable;
        var blocksAreDifferent = beforeBlock !== afterBlock;
        
        // glue case
        if ( openBefore && openAfter && blocksAreDifferent ) {
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'glue (with internal)' );
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();
          
          // for now, toss the after block (simplifies changes in one direction)
          this.unuseBlock( afterBlock );
          this.notePendingMoves( beforeBlock, interval.drawableAfter, drawableBeforeNextInterval );
          if ( !drawableBeforeNextInterval.nextDrawable ) {
            this.linkAfterDrawable( drawableBeforeNextInterval );
          }
          
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
        }
        // unglue case
        else if ( ( !openBefore || !openAfter ) && !blocksAreDifferent ) {
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'unglue' );
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();
          
          // for simplicity right now, we always create a fresh block (to avoid messing up reused blocks) after, and
          // always change everything after (instead of before), so we don't have to jump across multiple previous
          // change intervals
          var freshBlock = this.createBlock( drawableBeforeNextInterval.renderer, drawableBeforeNextInterval );
          this.blockOrderChanged = true; // needs to be done on block creation
          this.notePendingMoves( freshBlock, interval.drawableAfter, drawableBeforeNextInterval );
          if ( !drawableBeforeNextInterval.nextDrawable ) {
            this.linkAfterDrawable( drawableBeforeNextInterval );
          }
          
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
        }
        else {
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'no gluing needed' );
        }
      }
    },
    
    notePendingMoves: function( block, firstDrawable, lastDrawable ) {
      for ( var drawable = firstDrawable;; drawable = drawable.nextDrawable ) {
        assert && assert( !drawable.pendingAddition && !drawable.pendingRemoval,
                          'Moved drawables should be thought of as unchanged, and thus have nothing pending yet' );
        
        this.notePendingMove( drawable, block );
        if ( drawable === lastDrawable ) { break; }
      }
    },
    
    // If there is no currentBlock, we create one to match. Otherwise if the currentBlock is marked as 'unused' (i.e.
    // it is in the reusableBlocks array), we mark it as used so it won't me matched elsewhere.
    ensureUsedBlock: function( currentBlock, someIncludedDrawable ) {
      // if we have a matched block (or started with one)
      if ( currentBlock ) {
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'using existing block: ' + currentBlock.toString() );
        // since our currentBlock may be from reusableBlocks, we will need to mark it as used now.
        if ( !currentBlock.used ) {
          this.useBlock( currentBlock );
        }
      } else {
        // need to create one
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'searching for block' );
        currentBlock = this.getBlockForRenderer( someIncludedDrawable.renderer, someIncludedDrawable );
      }
      return currentBlock;
    },
    
    // NOTE: this doesn't handle hooking up the block linked list
    getBlockForRenderer: function( renderer, drawable ) {
      var block;
      
      // If it's not a DOM block, scan our reusable blocks for one with the same renderer.
      // If it's DOM, it should be processed correctly in reusableBlocks, and will never reach this point.
      if ( !Renderer.isDOM( renderer ) ) {
        // backwards scan, hopefully it's faster?
        for ( var i = this.reusableBlocks.length - 1; i >= 0; i-- ) {
          var tmpBlock = this.reusableBlocks[i];
          if ( tmpBlock.renderer === renderer ) {
            this.useBlockAtIndex( tmpBlock, i );
            block = tmpBlock;
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
    
    unuseBlock: function( block ) {
      block.used = false; // mark it as unused until we pull it out (so we can reuse, or quickly identify)
      this.reusableBlocks.push( block );
    },
    
    // removes a block from the list of reused blocks (done during matching)
    useBlock: function( block ) {
      this.useBlockAtIndex( block, _.indexOf( this.reusableBlocks, block ) );
    },
    
    useBlockAtIndex: function( block, index ) {
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'marking reusable block as used: ' + block.toString() + ' with renderer: ' + block.renderer );
      
      assert && assert( index >= 0 && this.reusableBlocks[index] === block, 'bad index for useBlockAtIndex: ' + index );
      
      assert && assert( block.used = false, 'Should be called on an unused (reusable) block' );
      
      // remove it
      this.reusableBlocks.splice( index, 1 );
      
      // mark it as used
      block.used = true;
    },
    
    // removes them from our domElement, and marks them for disposal
    removeUnusedBlocks: function( backbone ) {
      sceneryLog && sceneryLog.GreedyStitcher && this.reusableBlocks.length && sceneryLog.GreedyStitcher( 'removeUnusedBlocks' );
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();
      while ( this.reusableBlocks.length ) {
        this.markBlockForDisposal( this.reusableBlocks.pop() );
        this.blockOrderChanged = true;
      }
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();
    },
    
    linkBeforeDrawable: function( drawable ) {
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'link before ' + drawable.toString() );
      var beforeDrawable = drawable.previousDrawable;
      this.linkBlocks( beforeDrawable ? beforeDrawable.pendingParentDrawable : null,
                       drawable.pendingParentDrawable,
                       beforeDrawable,
                       drawable );
    },
    
    linkAfterDrawable: function( drawable ) {
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'link after ' + drawable.toString() );
      var afterDrawable = drawable.nextDrawable;
      this.linkBlocks( drawable.pendingParentDrawable,
                       afterDrawable ? afterDrawable.pendingParentDrawable : null,
                       drawable,
                       afterDrawable );
    },
    
    linkBlocks: function( beforeBlock, afterBlock, beforeDrawable, afterDrawable ) {
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( 'linking blocks: ' +
                                                                            ( beforeBlock ? ( beforeBlock.toString() + ' (' + beforeDrawable.toString() + ')' ) : 'null' ) +
                                                                            ' to ' +
                                                                            ( afterBlock ? ( afterBlock.toString() + ' (' + afterDrawable.toString() + ')' ) : 'null' ) );
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();
      
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
        this.markAfterBlock( beforeBlock, beforeDrawable );
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
        this.markBeforeBlock( afterBlock, afterDrawable );
      }
      
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();
    },
    
    processBlockLinkedList: function( backbone, firstBlock, lastBlock ) {
      // for now, just clear out the array first
      while ( backbone.blocks.length ) {
        backbone.blocks.pop();
      }
      
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( 'processBlockLinkedList: ' + firstBlock.toString() + ' to ' + lastBlock.toString() );
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();
      
      // leave the array as-is if there are no blocks
      if ( firstBlock ) {
        
        // rewrite it starting with the first block
        for ( var block = firstBlock;; block = block.nextBlock ) {
          sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( block.toString() );
          
          backbone.blocks.push( block );
          
          if ( block === lastBlock ) { break; }
        }
      }
      
      sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();
    }
  };
  
  var GreedyStitcher = scenery.GreedyStitcher = inherit( Stitcher, function GreedyStitcher() {
    // nothing done
  }, prototype );
  
  GreedyStitcher.stitchPrototype = prototype;
  
  return GreedyStitcher;
} );
