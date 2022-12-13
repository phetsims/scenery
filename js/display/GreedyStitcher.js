// Copyright 2014-2022, University of Colorado Boulder

/**
 * Stitcher that only rebuilds the parts necessary, and attempts greedy block matching as an optimization.
 *
 * Given a list of change intervals, our greedy stitcher breaks it down into 'sub-blocks' consisting of
 * drawables that are 'internal' to the change interval that all have the same renderer, and handles the
 * glue/unglue/matching situations in a greedy way by always using the first possible (allowing only one sweep
 * instead of multiple ones over the drawable linked list for this process).
 *
 * Conceptually, we break down drawables into groups that are 'internal' to each change interval (inside, not
 * including the un-changed ends), and 'external' (that are not internal to any intervals).
 *
 * For each interval, we first make sure that the next 'external' group of drawables has the proper blocks (for
 * instance, this can change with a glue/unglue operation, with processEdgeCases), then proceed to break the 'internal'
 * drawables into sub-blocks and process those with processSubBlock.
 *
 * Our stitcher has a list of blocks noted as 'reusable' that we use for two purposes:
 *   1. So that we can shift blocks to where they are needed, instead of removing (e.g.) an SVG block and
 *      creating another.
 *   2. So that blocks that are unused at the end of our stitch can be removed, and marked for disposal.
 * At the start of the stitch, we mark completely 'internal' blocks as reusable, so they can be shifted around as
 * necessary (used in a greedy way which may not be optimal). It's also possible during later phases for blocks that
 * also contain 'external' drawables to be marked as reusable, due to glue cases where before we needed multiple
 * blocks and now we only need one.
 *
 * We also use a linked-list of blocks during stitch operations (that then re-builds an array of blocks on any changes
 * after all stitching is done) for simplicity, and to avoid O(n^2) cases that would result from having to look up
 * indices in the block array during stitching.
 *
 * NOTE: Stitcher instances may be reused many times, even with different backbones. It should always release any
 * object references that it held after usage.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import cleanArray from '../../../phet-core/js/cleanArray.js';
import { Block, ChangeInterval, Drawable, Renderer, scenery, Stitcher } from '../imports.js';

// Returns whether the consecutive {Drawable}s 'a' and 'b' should be put into separate blocks
function hasGapBetweenDrawables( a, b ) {
  return a.renderer !== b.renderer || Renderer.isDOM( a.renderer ) || Renderer.isDOM( b.renderer );
}

// Whether the drawable and its previous sibling should be in the same block. Will be false if there is no sibling
function isOpenBefore( drawable ) {
  return drawable.previousDrawable !== null && !hasGapBetweenDrawables( drawable.previousDrawable, drawable );
}

// Whether the drawable and its next sibling should be in the same block. Will be false if there is no sibling
function isOpenAfter( drawable ) {
  return drawable.nextDrawable !== null && !hasGapBetweenDrawables( drawable, drawable.nextDrawable );
}

// If the change interval will contain any new (added) drawables
function intervalHasNewInternalDrawables( interval, firstStitchDrawable, lastStitchDrawable ) {
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

// If the change interval contained any drawables that are to be removed
function intervalHasOldInternalDrawables( interval, oldFirstStitchDrawable, oldLastStitchDrawable ) {
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

// Whether there are blocks that consist of drawables that are ALL internal to the {ChangeInterval} interval.
function intervalHasOldInternalBlocks( interval, firstStitchBlock, lastStitchBlock ) {
  const beforeBlock = interval.drawableBefore ? interval.drawableBefore.parentDrawable : null;
  const afterBlock = interval.drawableAfter ? interval.drawableAfter.parentDrawable : null;

  if ( beforeBlock && afterBlock && beforeBlock === afterBlock ) {
    return false;
  }

  if ( beforeBlock ) {
    return beforeBlock.nextBlock !== afterBlock; // OK for after to be null
  }
  else if ( afterBlock ) {
    return afterBlock.previousBlock !== beforeBlock; // OK for before to be null
  }
  else {
    return firstStitchBlock !== null;
  }
}

/**
 * Finds the furthest external drawable that:
 * (a) Before the next change interval (if we have a next change interval)
 * (b) Has the same renderer as the interval's drawableAfter
 */
function getLastCompatibleExternalDrawable( interval ) {
  const firstDrawable = interval.drawableAfter;

  if ( firstDrawable ) {
    const renderer = firstDrawable.renderer;

    // we stop our search before we reach this (null is acceptable), ensuring we don't go into the next change interval
    const cutoffDrawable = interval.nextChangeInterval ? interval.nextChangeInterval.drawableBefore.nextDrawable : null;

    let drawable = firstDrawable;

    while ( true ) { // eslint-disable-line no-constant-condition
      const nextDrawable = drawable.nextDrawable;

      // first comparison also does null check when necessary
      if ( nextDrawable !== cutoffDrawable && nextDrawable.renderer === renderer ) {
        drawable = nextDrawable;
      }
      else {
        break;
      }
    }

    return drawable;
  }
  else {
    return null; // with no drawableAfter, we don't have any external drawables after our interval
  }
}

class GreedyStitcher extends Stitcher {
  /**
   * Main stitch entry point, called directly from the backbone or cache. We are modifying our backbone's blocks and
   * their attached drawables.
   * @public
   *
   * The change-interval pair denotes a linked-list of change intervals that we will need to stitch across (they
   * contain drawables that need to be removed and added, and it may affect how we lay out blocks in the stacking
   * order).
   *
   * @param {BackboneDrawable} backbone
   * @param {Drawable|null} firstStitchDrawable
   * @param {Drawable|null} lastStitchDrawable
   * @param {Drawable|null} oldFirstStitchDrawable
   * @param {Drawable|null} oldLastStitchDrawable
   * @param {ChangeInterval} firstChangeInterval
   * @param {ChangeInterval} lastChangeInterval
   */
  stitch( backbone, firstStitchDrawable, lastStitchDrawable, oldFirstStitchDrawable, oldLastStitchDrawable, firstChangeInterval, lastChangeInterval ) {
    // required call to the Stitcher interface (see Stitcher.initialize()).
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

    // To properly handle glue/unglue situations with external blocks (ones that aren't reusable after phase 1),
    // we need some extra tracking for our inner sub-block edge case loop.
    this.blockWasAdded = false; // we need to know if a previously-existing block was added, and remove it otherwise.

    let interval;

    // record current first/last drawables for the entire backbone
    this.recordBackboneBoundaries();

    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( 'phase 1: old linked list' );
    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();

    // Handle pending removal of old blocks/drawables. First, we need to mark all 'internal' drawables with
    // notePendingRemoval(), so that if they aren't added back in this backbone, that they are removed from their
    // old block. Note that later we will add the ones that stay on this backbone, so that they only either change
    // blocks, or stay on the same block.
    if ( backbone.blocks.length ) {
      const veryFirstBlock = backbone.blocks[ 0 ];
      const veryLastBlock = backbone.blocks[ backbone.blocks.length - 1 ];

      for ( interval = firstChangeInterval; interval !== null; interval = interval.nextChangeInterval ) {
        assert && assert( !interval.isEmpty(), 'We now guarantee that the intervals are non-empty' );

        // First, we need to mark all old 'internal' drawables with notePendingRemoval(), so that if they aren't added
        // back in this backbone, that they are removed from their old block. Note that later we will add the ones
        // that stay on this backbone, so that they only either change blocks, or stay on the same block.
        if ( intervalHasOldInternalDrawables( interval, oldFirstStitchDrawable, oldLastStitchDrawable ) ) {
          const firstRemoval = interval.drawableBefore ?
                               interval.drawableBefore.oldNextDrawable :
                               oldFirstStitchDrawable;
          const lastRemoval = interval.drawableAfter ?
                              interval.drawableAfter.oldPreviousDrawable :
                              oldLastStitchDrawable;

          // drawable iteration on the 'old' linked list
          for ( let removedDrawable = firstRemoval; ; removedDrawable = removedDrawable.oldNextDrawable ) {
            this.notePendingRemoval( removedDrawable );
            if ( removedDrawable === lastRemoval ) { break; }
          }
        }

        // Blocks totally contained within the change interval are marked as reusable (doesn't include end blocks).
        if ( intervalHasOldInternalBlocks( interval, veryFirstBlock, veryLastBlock ) ) {
          const firstBlock = interval.drawableBefore === null ? backbone.blocks[ 0 ] : interval.drawableBefore.parentDrawable.nextBlock;
          const lastBlock = interval.drawableAfter === null ? backbone.blocks[ backbone.blocks.length - 1 ] : interval.drawableAfter.parentDrawable.previousBlock;

          for ( let markedBlock = firstBlock; ; markedBlock = markedBlock.nextBlock ) {
            this.unuseBlock( markedBlock );
            if ( markedBlock === lastBlock ) { break; }
          }
        }
      }
    }

    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();

    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( 'phase 2: new linked list' );
    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();

    // Don't process the single interval left if we aren't left with any drawables (thus left with no blocks)
    if ( firstStitchDrawable ) {
      for ( interval = firstChangeInterval; interval !== null; interval = interval.nextChangeInterval ) {
        this.processInterval( backbone, interval, firstStitchDrawable, lastStitchDrawable );
      }
    }

    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();

    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( 'phase 3: cleanup' );
    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();

    // Anything in our 'reusable' blocks array should be removed from our DOM and marked for disposal.
    this.removeUnusedBlocks();

    // Fire off notifyInterval calls to blocks if their boundaries (first/last drawables) have changed. This is
    // a necessary call since we used markBeforeBlock/markAfterBlock to record block boundaries as we went along.
    // We don't want to do this synchronously, because then you could update a block's boundaries multiple times,
    // which may be expensive.
    this.updateBlockIntervals();

    if ( firstStitchDrawable === null ) {
      // i.e. clear our blocks array
      this.useNoBlocks();
    }
    else if ( this.blockOrderChanged ) {
      // Rebuild our blocks array from the linked list format we used for recording our changes (avoids O(n^2)
      // situations since we don't need to do array index lookups while making changes, but only at the end).
      this.processBlockLinkedList( backbone, firstStitchDrawable.pendingParentDrawable, lastStitchDrawable.pendingParentDrawable );

      // Actually reindex the DOM elements of the blocks (changing as necessary)
      this.reindex();
    }

    // required call to the Stitcher interface (see Stitcher.clean()).
    this.clean();

    // release the references we made in this type
    cleanArray( this.reusableBlocks );

    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();
  }

  /**
   * Does the main bulk of the work for each change interval.
   * @private
   *
   * @param {BackboneDrawable} backbone
   * @param {ChangeInterval} interval
   * @param {Drawable|null} firstStitchDrawable
   * @param {Drawable|null} lastStitchDrawable
   */
  processInterval( backbone, interval, firstStitchDrawable, lastStitchDrawable ) {
    assert && assert( interval instanceof ChangeInterval );
    assert && assert( firstStitchDrawable instanceof Drawable, 'We assume we have a non-null remaining section' );
    assert && assert( lastStitchDrawable instanceof Drawable, 'We assume we have a non-null remaining section' );
    assert && assert( !interval.isEmpty(), 'We now guarantee that the intervals are non-empty' );

    sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `interval: ${
      interval.drawableBefore ? interval.drawableBefore.toString() : 'null'
    } to ${
      interval.drawableAfter ? interval.drawableAfter.toString() : 'null'}` );
    sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();

    // check if our interval removes everything, we may need a glue
    if ( !intervalHasNewInternalDrawables( interval, firstStitchDrawable, lastStitchDrawable ) ) {
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'no current internal drawables in interval' );

      // separate if, last condition above would cause issues with the normal operation branch
      if ( interval.drawableBefore && interval.drawableAfter ) {
        assert && assert( interval.drawableBefore.nextDrawable === interval.drawableAfter );

        // if we removed everything (no new internal drawables), our drawableBefore is open 'after', if our
        // drawableAfter is open 'before' since they are siblings (only one flag needed).
        const isOpen = !hasGapBetweenDrawables( interval.drawableBefore, interval.drawableAfter );

        // handle glue/unglue or any other 'external' changes
        this.processEdgeCases( interval, isOpen, isOpen );
      }

      if ( interval.drawableBefore && !isOpenAfter( interval.drawableBefore ) ) {
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'closed-after collapsed link:' );
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();
        this.linkAfterDrawable( interval.drawableBefore );
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
      }
      else if ( interval.drawableAfter && !isOpenBefore( interval.drawableAfter ) ) {
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'closed-before collapsed link:' );
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();
        this.linkBeforeDrawable( interval.drawableAfter );
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
      }
    }
    // otherwise normal operation
    else {
      let drawable = interval.drawableBefore ? interval.drawableBefore.nextDrawable : firstStitchDrawable;

      // if we have any current drawables at all
      let subBlockFirstDrawable = null;
      let matchedBlock = null;
      let isFirst = true;

      // separate our new-drawable linked-list into sub-blocks that we will process individually
      while ( true ) { // eslint-disable-line no-constant-condition
        const nextDrawable = drawable.nextDrawable;
        const isLast = nextDrawable === interval.drawableAfter;

        assert && assert( nextDrawable !== null || isLast, 'If our nextDrawable is null, isLast must be true' );

        if ( !subBlockFirstDrawable ) {
          subBlockFirstDrawable = drawable;
        }

        // See if any of our 'new' drawables were part of a block that we've marked as reusable. If this is the case,
        // we'll greedily try to use this block for matching if possible (ignoring the other potential matches for
        // other drawables after in the same sub-block).
        if ( matchedBlock === null && drawable.parentDrawable && !drawable.parentDrawable.used && drawable.backbone === backbone &&
             drawable.parentDrawable.parentDrawable === backbone ) {
          matchedBlock = drawable.parentDrawable;
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `matching at ${drawable.toString()} with ${matchedBlock}` );
        }

        if ( isLast || hasGapBetweenDrawables( drawable, nextDrawable ) ) {
          if ( isFirst ) {
            // we'll handle any glue/unglue at the start, so every processSubBlock can be set correctly.
            this.processEdgeCases( interval, isOpenBefore( subBlockFirstDrawable ), isOpenAfter( drawable ) );
          }

          // do the necessary work for each sub-block (adding drawables, linking, using matched blocks)
          this.processSubBlock( interval, subBlockFirstDrawable, drawable, matchedBlock, isFirst, isLast );

          subBlockFirstDrawable = null;
          matchedBlock = null;
          isFirst = false;
        }

        if ( isLast ) {
          break;
        }
        else {
          drawable = nextDrawable;
        }
      }
    }


    sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
  }

  /**
   * @private
   *
   * @param {ChangeInterval} interval
   * @param {Drawable} firstDrawable - for the specific sub-block
   * @param {Drawable} lastDrawable - for the specific sub-block
   * @param {Block} matchedBlock
   * @param {boolean} isFirst
   * @param {boolean} isLast
   */
  processSubBlock( interval, firstDrawable, lastDrawable, matchedBlock, isFirst, isLast ) {
    sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose(
      `sub-block: ${firstDrawable.toString()} to ${lastDrawable.toString()} ${
        matchedBlock ? `with matched: ${matchedBlock.toString()}` : 'with no match'}` );
    sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();

    const openBefore = isOpenBefore( firstDrawable );
    const openAfter = isOpenAfter( lastDrawable );

    assert && assert( !openBefore || isFirst, 'openBefore implies isFirst' );
    assert && assert( !openAfter || isLast, 'openAfter implies isLast' );

    assert && assert( !openBefore || !openAfter || firstDrawable.previousDrawable.pendingParentDrawable === lastDrawable.nextDrawable.pendingParentDrawable,
      'If we would use both the before and after blocks, make sure any gluing ' );

    // if our sub-block gets combined into the previous block, use its block instead of any match-scanned blocks
    if ( openBefore ) {
      matchedBlock = firstDrawable.previousDrawable.pendingParentDrawable;
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `combining into before block: ${matchedBlock.toString()}` );
    }

    // if our sub-block gets combined into the next block, use its block instead of any match-scanned blocks
    if ( openAfter ) {
      matchedBlock = lastDrawable.nextDrawable.pendingParentDrawable;
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `combining into after block: ${matchedBlock.toString()}` );
    }

    // create a block if matchedBlock is null, otherwise mark it as used (if it is in reusableBlocks)
    matchedBlock = this.ensureUsedBlock( matchedBlock, firstDrawable );

    // add internal drawables
    for ( let drawable = firstDrawable; ; drawable = drawable.nextDrawable ) {
      this.notePendingAddition( drawable, matchedBlock );
      if ( drawable === lastDrawable ) { break; }
    }

    // link our blocks (and set pending block boundaries) as needed. assumes glue/unglue has already been performed
    if ( !openBefore ) {
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'closed-before link:' );
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();
      this.linkBeforeDrawable( firstDrawable );
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
    }
    if ( isLast && !openAfter ) {
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'last closed-after link:' );
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();
      this.linkAfterDrawable( lastDrawable );
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
    }

    sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
  }

  /**
   * firstDrawable and lastDrawable refer to the specific sub-block (if it exists), isLast refers to if it's the
   * last sub-block
   * @private
   *
   * @param {ChangeInterval} interval
   * @param {boolean} openBefore
   * @param {boolean} openAfter
   */
  processEdgeCases( interval, openBefore, openAfter ) {
    // this test passes for glue and unglue cases
    if ( interval.drawableBefore !== null && interval.drawableAfter !== null ) {
      const beforeBlock = interval.drawableBefore.pendingParentDrawable;
      const afterBlock = interval.drawableAfter.pendingParentDrawable;
      const nextAfterBlock = ( interval.nextChangeInterval && interval.nextChangeInterval.drawableAfter ) ?
                             interval.nextChangeInterval.drawableAfter.pendingParentDrawable :
                             null;

      // Since we want to remove any afterBlock at the end of its run if we don't have blockWasAdded set, this check
      // is necessary to see if we have already used this specific block.
      // Otherwise, we would remove our (potentially very-first) block when it has already been used externally.
      if ( beforeBlock === afterBlock ) {
        this.blockWasAdded = true;
      }

      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose(
        `edge case: ${
          openBefore ? 'open-before ' : ''
        }${openAfter ? 'open-after ' : ''
        }${beforeBlock.toString()} to ${afterBlock.toString()}` );
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();

      // deciding what new block should be used for the external group of drawables after our change interval
      let newAfterBlock;
      // if we have no gaps/boundaries, we should not have two different blocks
      if ( openBefore && openAfter ) {
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `glue using ${beforeBlock.toString()}` );
        newAfterBlock = beforeBlock;
      }
      else {
        // if we can't use our afterBlock, since it was used before, or wouldn't create a split
        if ( this.blockWasAdded || beforeBlock === afterBlock ) {
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'split with fresh block' );
          // for simplicity right now, we always create a fresh block (to avoid messing up reused blocks) after, and
          // always change everything after (instead of before), so we don't have to jump across multiple previous
          // change intervals
          newAfterBlock = this.createBlock( interval.drawableAfter.renderer, interval.drawableAfter );
          this.blockOrderChanged = true; // needs to be done on block creation
        }
        // otherwise we can use our after block
        else {
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `split with same afterBlock ${afterBlock.toString()}` );
          newAfterBlock = afterBlock;
        }
      }

      // If we didn't change our block, mark it as added so we don't remove it.
      if ( afterBlock === newAfterBlock ) {
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'no externals change here (blockWasAdded => true)' );
        this.blockWasAdded = true;
      }
        // Otherwise if we changed the block, move over only the external drawables between this interval and the next
      // interval.
      else {
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'moving externals' );
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.push();
        this.changeExternals( interval, newAfterBlock );
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
      }

      // If the next interval's old afterBlock isn't the same as our old afterBlock, we need to make our decision
      // about whether to mark our old afterBlock as reusable, or whether it was used.
      if ( nextAfterBlock !== afterBlock ) {
        sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'end of afterBlock stretch' );

        // If our block wasn't added yet, it wouldn't ever be added later naturally (so we mark it as reusable).
        if ( !this.blockWasAdded ) {
          sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `unusing ${afterBlock.toString()}` );
          this.unuseBlock( afterBlock );
        }
        this.blockWasAdded = false;
      }

      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.pop();
    }
  }

  /**
   * Marks all 'external' drawables from the end (drawableAfter) of the {ChangeInterval} interval to either the end
   * of their old block or the drawableAfter of the next interval (whichever is sooner) as being needed to be moved to
   * our {Block} newBlock. The next processInterval will both handle the drawables inside that next interval, and
   * will be responsible for the 'external' drawables after that.
   * @private
   *
   * @param {ChangeInterval} interval
   * @param {Block} newBlock
   */
  changeExternals( interval, newBlock ) {
    const lastExternalDrawable = getLastCompatibleExternalDrawable( interval );
    this.notePendingMoves( newBlock, interval.drawableAfter, lastExternalDrawable );

    // If we didn't make it all the way to the next change interval's drawableBefore (there was another block
    // starting before the next interval), we need to link our new block to that next block.
    if ( !interval.nextChangeInterval || interval.nextChangeInterval.drawableBefore !== lastExternalDrawable ) {
      this.linkAfterDrawable( lastExternalDrawable );
    }
  }

  /**
   * Given a {Drawable} firstDrawable and {Drawable} lastDrawable, we mark all drawables in-between (inclusively) as
   * needing to be moved to our {Block} newBlock. This should only be called on external drawables, and should only
   * occur as needed with glue/unglue cases in the stitch.
   * @private
   *
   * @param {Block} newBlock
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   */
  notePendingMoves( newBlock, firstDrawable, lastDrawable ) {
    for ( let drawable = firstDrawable; ; drawable = drawable.nextDrawable ) {
      assert && assert( !drawable.pendingAddition && !drawable.pendingRemoval,
        'Moved drawables should be thought of as unchanged, and thus have nothing pending yet' );

      this.notePendingMove( drawable, newBlock );
      if ( drawable === lastDrawable ) { break; }
    }
  }

  /**
   * If there is no currentBlock, we create one to match. Otherwise if the currentBlock is marked as 'unused' (i.e.
   * it is in the reusableBlocks array), we mark it as used so it won't me matched elsewhere.
   * @private
   *
   * @param {Block} currentBlock
   * @param {Drawable} someIncludedDrawable
   * @returns {Block}
   */
  ensureUsedBlock( currentBlock, someIncludedDrawable ) {
    // if we have a matched block (or started with one)
    if ( currentBlock ) {
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `using existing block: ${currentBlock.toString()}` );
      // since our currentBlock may be from reusableBlocks, we will need to mark it as used now.
      if ( !currentBlock.used ) {
        this.useBlock( currentBlock );
      }
    }
    else {
      // need to create one
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( 'searching for block' );
      currentBlock = this.getBlockForRenderer( someIncludedDrawable.renderer, someIncludedDrawable );
    }
    return currentBlock;
  }

  /**
   * Attemps to find an unused block with the same renderer if possible, otherwise creates a
   * compatible block.
   * @private
   *
   * NOTE: this doesn't handle hooking up the block linked list
   *
   * @param {number} renderer
   * @param {Drawable} drawable
   * @returns {Block}
   */
  getBlockForRenderer( renderer, drawable ) {
    let block;

    // If it's not a DOM block, scan our reusable blocks for one with the same renderer.
    // If it's DOM, it should be processed correctly in reusableBlocks, and will never reach this point.
    if ( !Renderer.isDOM( renderer ) ) {
      // backwards scan, hopefully it's faster?
      for ( let i = this.reusableBlocks.length - 1; i >= 0; i-- ) {
        const tmpBlock = this.reusableBlocks[ i ];
        assert && assert( !tmpBlock.used );
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
  }

  /**
   * Marks a block as unused, moving it to the reusableBlocks array.
   * @private
   *
   * @param {Block} block
   */
  unuseBlock( block ) {
    if ( block.used ) {
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `unusing block: ${block.toString()}` );
      block.used = false; // mark it as unused until we pull it out (so we can reuse, or quickly identify)
      this.reusableBlocks.push( block );
    }
    else {
      sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `not using already-unused block: ${block.toString()}` );
    }
  }

  /**
   * Removes a block from the list of reused blocks (done during matching)
   * @private
   *
   * @param {Block} block
   */
  useBlock( block ) {
    this.useBlockAtIndex( block, _.indexOf( this.reusableBlocks, block ) );
  }

  /**
   * @private
   *
   * @param {Block} block
   * @param {number} index
   */
  useBlockAtIndex( block, index ) {
    sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `using reusable block: ${block.toString()} with renderer: ${block.renderer}` );

    assert && assert( index >= 0 && this.reusableBlocks[ index ] === block, `bad index for useBlockAtIndex: ${index}` );

    assert && assert( !block.used, 'Should be called on an unused (reusable) block' );

    // remove it
    this.reusableBlocks.splice( index, 1 );

    // mark it as used
    block.used = true;
  }

  /**
   * Removes all of our unused blocks from our domElement, and marks them for disposal.
   * @private
   */
  removeUnusedBlocks() {
    sceneryLog && sceneryLog.GreedyStitcher && this.reusableBlocks.length && sceneryLog.GreedyStitcher( 'removeUnusedBlocks' );
    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();
    while ( this.reusableBlocks.length ) {
      const block = this.reusableBlocks.pop();
      this.markBlockForDisposal( block );
      this.blockOrderChanged = true;
    }
    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();
  }

  /**
   * Links blocks before a drawable (whether it is the first drawable or not)
   * @private
   *
   * @param {Drawable} drawable
   */
  linkBeforeDrawable( drawable ) {
    sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `link before ${drawable.toString()}` );
    const beforeDrawable = drawable.previousDrawable;
    this.linkBlocks( beforeDrawable ? beforeDrawable.pendingParentDrawable : null,
      drawable.pendingParentDrawable,
      beforeDrawable,
      drawable );
  }


  /**
   * links blocks after a drawable (whether it is the last drawable or not)
   * @private
   *
   * @param {Drawable} drawable
   */
  linkAfterDrawable( drawable ) {
    sceneryLog && sceneryLog.GreedyVerbose && sceneryLog.GreedyVerbose( `link after ${drawable.toString()}` );
    const afterDrawable = drawable.nextDrawable;
    this.linkBlocks( drawable.pendingParentDrawable,
      afterDrawable ? afterDrawable.pendingParentDrawable : null,
      drawable,
      afterDrawable );
  }

  /**
   * Called to mark a boundary between blocks, or at the end of our list of blocks (one block/drawable pair being
   * null notes that it is the start/end, and there is no previous/next block).
   * This updates the block linked-list as necessary (noting changes when they occur) so that we can rebuild an array
   * at the end of the stitch, avoiding O(n^2) issues if we were to do block-array-index lookups during linking
   * operations (this results in linear time for blocks).
   * It also marks block boundaries as dirty when necessary, so that we can make one pass through with
   * updateBlockIntervals() that updates all of the block's boundaries (avoiding more than one update per block per
   * frame).
   * @private
   *
   * @param {Block|null} beforeBlock
   * @param {Block|null} afterBlock
   * @param {Drawable|null} beforeDrawable
   * @param {Drawable|null} afterDrawable
   */
  linkBlocks( beforeBlock, afterBlock, beforeDrawable, afterDrawable ) {
    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( `linking blocks: ${
      beforeBlock ? ( `${beforeBlock.toString()} (${beforeDrawable.toString()})` ) : 'null'
    } to ${
      afterBlock ? ( `${afterBlock.toString()} (${afterDrawable.toString()})` ) : 'null'}` );
    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();

    assert && assert( ( beforeBlock === null && beforeDrawable === null ) ||
                      ( beforeBlock instanceof Block && beforeDrawable instanceof Drawable ) );
    assert && assert( ( afterBlock === null && afterDrawable === null ) ||
                      ( afterBlock instanceof Block && afterDrawable instanceof Drawable ) );

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
  }

  /**
   * Rebuilds the backbone's block array from our linked-list data.
   * @private
   *
   * @param {BackboneDrawable} backbone
   * @param {Block|null} firstBlock
   * @param {Block|null} lastBlock
   */
  processBlockLinkedList( backbone, firstBlock, lastBlock ) {
    // for now, just clear out the array first
    while ( backbone.blocks.length ) {
      backbone.blocks.pop();
    }

    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( `processBlockLinkedList: ${firstBlock.toString()} to ${lastBlock.toString()}` );
    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.push();

    // leave the array as-is if there are no blocks
    if ( firstBlock ) {

      // rewrite it starting with the first block
      for ( let block = firstBlock; ; block = block.nextBlock ) {
        sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.GreedyStitcher( block.toString() );

        backbone.blocks.push( block );

        if ( block === lastBlock ) { break; }
      }
    }

    sceneryLog && sceneryLog.GreedyStitcher && sceneryLog.pop();
  }
}

scenery.register( 'GreedyStitcher', GreedyStitcher );
export default GreedyStitcher;