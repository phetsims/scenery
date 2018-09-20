// Copyright 2014-2016, University of Colorado Boulder


/**
 * Abstract base type (and API) for stitching implementations. Stitching is:
 * A method of updating the blocks for a backbone (the changes from the previous frame to the current frame), and
 * setting up the drawables to be attached/detached from blocks. At a high level:
 *   - We have an ordered list of blocks displayed in the last frame.
 *   - We have an ordered list of drawables displayed in the last frame (and what block they are part of).
 *   - We have an ordered list of drawables that will be displayed in the next frame (and whether they were part of our
 *     backbone, and if so what block they were in).
 *   - We need to efficiently create/dispose required blocks, add/remove drawables from blocks, notify blocks of their
 *     drawable range, and ensure blocks are displayed back-to-front.
 *
 * Since stitching usually only involves one or a few small changes (except for on sim initialization), the stitch
 * method is provided with a list of intervals that were (potentially) changed. This consists of a linked-list of
 * intervals (it is constructed during recursion through a tree that skips known-unchanged subtrees). The intervals
 * are completely disjoint (don't overlap, and aren't adjacent - there is at least one drawable that is unchanged
 * in-between change intervals).
 *
 * Assumes the same object instance will be reused multiple times, possibly for different backbones.
 *
 * Any stitcher implementations should always call initialize() first and clean() at the end, so that we can set up
 * and then clean up any object references (allowing them to be garbage-collected or pooled more safely).
 *
 * Stitcher responsibilities:
 *   1. Blocks used in the previous frame but not used in the current frame (no drawables, not attached) should be
 *      marked for disposal.
 *   2. Blocks should be created as necessary.
 *   3. If a changed drawable is removed from a block, it should have notePendingRemoval called on it.
 *   4. If a changed drawable is added to a block, it should have notePendingAddition called on it.
 *   5. If an unchanged drawable is to have a block change, it should have notePendingMove called on it.
 *   6. New blocks should be added to the DOM (appendChild presumably)
 *   7. Removed blocks should be removed from the DOM (removeChild)
 *      NOTE: check for child-parent relationship, since DOM blocks (wrappers) may have been
 *      added to the DOM elsewhere in another backbone's stitch already (which in the DOM
 *      automatically removes it from our backbone's div)
 *   8. If a block's first or last drawable changes, it should have notifyInterval called on it.
 *   9. At the end of the stitch, the backbone should have a way of iterating over its blocks in order (preferably an
 *      Array for fast repaint iteration)
 *   10. New blocks should have setBlockBackbone( backbone ) called on them
 *   11. Blocks with any drawable change should have backbone.markDirtyDrawable( block ) called so it can be visited
 *       in the repaint phase.
 *   12. Blocks should have z-indices set in the proper stacking order (back to front), using backbone.reindexBlocks()
 *       or equivalent (it tries to change as few z-indices as possible).
 *
 * Stitcher desired behavior and optimizations:
 *   1. Reuse blocks of the same renderer type, instead of removing one and creating another.
 *   2. Minimize (as much as is possible) how many drawables are added and removed from blocks (try not to remove 1000
 *      drawables from A and add them to B if we could instead just add/remove 5 drawables from C to D)
 *   3. No more DOM manipulation than necessary
 *   4. Optimize first for "one or a few small change intervals" that only cause local changes (no blocks created,
 *      removed or reordered). It would be ideal to do this very quickly, so it could be done every frame in
 *      simulations.
 *
 * Current constraints:
 *   1. DOM drawables should be paired with exactly one block (basically a wrapper, they are inserted directly into the
 *      DOM, and a DOM block should only ever be given the same drawable.
 *   2. Otherwise, consecutive drawables with the same renderer should be part of the same block. In the future we will
 *      want to allow "gaps" to form between (if something with a different renderer gets added and removed a lot
 *      in-between), but we'll need to figure out performance-sensitive flags to indicate when this needs to not be
 *      done (opacity and types of blending require no gaps between same-renderer drawables).
 *
 * Gluing: consequences of "no gaps"
 * There are two (important) implications:
 * Gluing
 *   If we have the following blocks:
 *     … A (SVG), B (Canvas), C (SVG) ...
 *   and all drawables for for B are removed, the following would be invalid ("has a gap"):
 *     … A (SVG), C (SVG) …
 *   so we need to glue them together, usually either resulting in:
 *     … A (SVG) …
 *   or
 *     … C (SVG) …
 *   with A or C including all of the drawables that were in A and C.
 *   More generally:
 *     If a change interval used to have its before/after (unchanged) drawables on two
 *     different blocks and for the current frame there will be no blocks in-between,
 *     we will need to "glue".
 *   Additionally, note the case:
 *     … A (SVG), B (Canvas), C (DOM), D (SVG), E (Canvas), F (SVG).
 *   If B,C,E are all removed, the results of A,D,F will have to all be combined into one layer
 * Un-gluing
 *   If we have the following drawables, all part of one block:
 *     … a (svg), b (svg) …
 *   and we insert a drawable with a different renderer:
 *     … a (svg), c (canvas), b (svg) ...
 *   we will need to split them into to SVG blocks
 *   More generally:
 *     If a change interval used to have its before/after (unchanged) drawables included
 *     in the same block, and the current frame requires a block to be inserted
 *     in-between, we will need to "un-glue".
 * These consequences mean that "unchanged" drawables (outside of change intervals) may need to have their block changed
 * (with notePendingMove). For performance, please consider which "end" should keep its drawables (the other end's
 * drawables will ALL have to be added/removed, which can be a major performance loss if we choose the wrong one).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var CanvasBlock = require( 'SCENERY/display/CanvasBlock' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var DOMBlock = require( 'SCENERY/display/DOMBlock' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var SVGBlock = require( 'SCENERY/display/SVGBlock' );
  var WebGLBlock = require( 'SCENERY/display/WebGLBlock' );

  function Stitcher( display, renderer ) {
    throw new Error( 'We are too abstract for that!' );
  }

  scenery.register( 'Stitcher', Stitcher );

  inherit( Object, Stitcher, {
    // Main stitch entry point, called directly from the backbone or cache. We are modifying our backbone's blocks and
    // their attached drawables.
    // @param {Drawable | null} firstStitchDrawable: What our backbone's first drawable will be after this stitch
    // @param {Drawable | null} lastStitchDrawable: What our backbone's last drawable will be after this stitch
    // @param {Drawable | null} oldFirstStitchDrawable: What our backbone's first drawable was before this stitch
    // @param {Drawable | null} oldLastStitchDrawable: What our backbone's last drawable was before this stitch
    // @param {ChangeInterval} firstChangeInterval: The first change interval of our interval linked-list
    // @param {ChangeInterval} lastChangeInterval: The last change interval of our interval linked-list
    // The change-interval pair denotes a linked-list of change intervals that we will need to stitch across (they
    // contain drawables that need to be removed and added, and it may affect how we lay out blocks in the stacking
    // order).
    initialize: function( backbone, firstDrawable, lastDrawable, oldFirstDrawable, oldLastDrawable, firstChangeInterval, lastChangeInterval ) {
      assert && assert( firstChangeInterval && lastChangeInterval, 'We are guaranteed at least one change interval' );
      assert && assert( !firstDrawable || firstDrawable.previousDrawable === null,
        'End boundary of drawable linked list should link to null' );
      assert && assert( !lastDrawable || lastDrawable.nextDrawable === null,
        'End boundary of drawable linked list should link to null' );

      if ( sceneryLog && sceneryLog.Stitch ) {
        sceneryLog.Stitch( 'stitch ' + backbone.toString() +
                           ' first:' + ( firstDrawable ? firstDrawable.toString() : 'null' ) +
                           ' last:' + ( lastDrawable ? lastDrawable.toString() : 'null' ) +
                           ' oldFirst:' + ( oldFirstDrawable ? oldFirstDrawable.toString() : 'null' ) +
                           ' oldLast:' + ( oldLastDrawable ? oldLastDrawable.toString() : 'null' ) );
        sceneryLog.push();
      }
      if ( sceneryLog && sceneryLog.StitchDrawables ) {
        sceneryLog.StitchDrawables( 'Before:' );
        sceneryLog.push();
        Stitcher.debugDrawables( oldFirstDrawable, oldLastDrawable, firstChangeInterval, lastChangeInterval, false );
        sceneryLog.pop();

        sceneryLog.StitchDrawables( 'After:' );
        sceneryLog.push();
        Stitcher.debugDrawables( firstDrawable, lastDrawable, firstChangeInterval, lastChangeInterval, true );
        sceneryLog.pop();
      }

      this.backbone = backbone;
      this.firstDrawable = firstDrawable;
      this.lastDrawable = lastDrawable;

      // list of blocks that have their pendingFirstDrawable or pendingLastDrawable set, and need updateInterval() called
      this.touchedBlocks = cleanArray( this.touchedBlocks );

      if ( assertSlow ) {
        assertSlow( !this.initialized, 'We should not be already initialized (clean should be called)' );
        this.initialized = true;
        this.reindexed = false;

        this.pendingAdditions = [];
        this.pendingRemovals = [];
        this.pendingMoves = [];
        this.createdBlocks = [];
        this.disposedBlocks = [];
        this.intervalsNotified = [];
        this.boundariesRecorded = false;

        this.previousBlocks = backbone.blocks.slice( 0 ); // copy of previous blocks
      }
    },

    // Removes object references
    clean: function() {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'clean' );
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( '-----------------------------------' );

      if ( assertSlow ) {
        this.auditStitch();

        this.initialized = false;
      }

      this.backbone = null;
      this.firstDrawable = null;
      this.lastDrawable = null;

      sceneryLog && sceneryLog.Stitch && sceneryLog.pop();
    },

    // Writes the first/last drawables for the entire backbone into its memory. We want to wait to do this until we have
    // read from its previous values.
    recordBackboneBoundaries: function() {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'recording backbone boundaries: ' +
                                                            ( this.firstDrawable ? this.firstDrawable.toString() : 'null' ) +
                                                            ' to ' +
                                                            ( this.lastDrawable ? this.lastDrawable.toString() : 'null' ) );
      this.backbone.previousFirstDrawable = this.firstDrawable;
      this.backbone.previousLastDrawable = this.lastDrawable;

      if ( assertSlow ) {
        this.boundariesRecorded = true;
      }
    },

    // Records that this {Drawable} drawable should be added/moved to the {Block} at a later time
    notePendingAddition: function( drawable, block ) {
      assert && assert( drawable.renderer === block.renderer );

      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'pending add: ' + drawable.toString() + ' to ' + block.toString() );
      sceneryLog && sceneryLog.Stitch && sceneryLog.push();

      drawable.notePendingAddition( this.backbone.display, block, this.backbone );

      if ( assertSlow ) {
        this.pendingAdditions.push( {
          drawable: drawable,
          block: block
        } );
      }

      sceneryLog && sceneryLog.Stitch && sceneryLog.pop();
    },

    // Records that this {Drawable} drawable should be moved to the {Block} at a later time (called only on external
    // drawables). notePendingAddition and notePendingRemoval should not be called on a drawable that had
    // notePendingMove called on it during the same stitch, and vice versa.
    notePendingMove: function( drawable, block ) {
      assert && assert( drawable.renderer === block.renderer );

      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'pending move: ' + drawable.toString() + ' to ' + block.toString() );
      sceneryLog && sceneryLog.Stitch && sceneryLog.push();

      drawable.notePendingMove( this.backbone.display, block );

      if ( assertSlow ) {
        this.pendingMoves.push( {
          drawable: drawable,
          block: block
        } );
      }

      sceneryLog && sceneryLog.Stitch && sceneryLog.pop();
    },

    // Records that this {Drawable} drawable should be removed/moved from the {Block} at a later time
    notePendingRemoval: function( drawable ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'pending remove: ' + drawable.toString() );
      sceneryLog && sceneryLog.Stitch && sceneryLog.push();

      drawable.notePendingRemoval( this.backbone.display );

      if ( assertSlow ) {
        this.pendingRemovals.push( {
          drawable: drawable
        } );
      }

      sceneryLog && sceneryLog.Stitch && sceneryLog.pop();
    },

    // Records that this {Block} block should be disposed at a later time. It should not be in the blocks array at the
    // end of the stitch.
    markBlockForDisposal: function( block ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'block for disposal: ' + block.toString() );
      sceneryLog && sceneryLog.Stitch && sceneryLog.push();

      //TODO: PERFORMANCE: does this cause reflows / style calculation
      if ( block.domElement.parentNode === this.backbone.domElement ) {
        // guarded, since we may have a (new) child drawable add it before we can remove it
        this.backbone.domElement.removeChild( block.domElement );
      }
      block.markForDisposal( this.backbone.display );

      if ( assertSlow ) {
        this.disposedBlocks.push( {
          block: block
        } );
      }

      sceneryLog && sceneryLog.Stitch && sceneryLog.pop();
    },

    removeAllBlocks: function() {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'marking all blocks for disposal (count ' + this.backbone.blocks.length + ')' );
      sceneryLog && sceneryLog.Stitch && sceneryLog.push();

      while ( this.backbone.blocks.length ) {
        var block = this.backbone.blocks[ 0 ];

        this.removeBlock( block );
        this.markBlockForDisposal( block );
      }

      sceneryLog && sceneryLog.Stitch && sceneryLog.pop();
    },

    // Immediately notify a block of its first/last drawable.
    notifyInterval: function( block, firstDrawable, lastDrawable ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'notify interval: ' + block.toString() + ' ' +
                                                            firstDrawable.toString() + ' to ' + lastDrawable.toString() );
      sceneryLog && sceneryLog.Stitch && sceneryLog.push();

      block.notifyInterval( firstDrawable, lastDrawable );

      // mark it dirty, since its drawables probably changed?
      //OHTWO TODO: is this necessary? What is this doing?
      this.backbone.markDirtyDrawable( block );

      if ( assertSlow ) {
        this.intervalsNotified.push( {
          block: block,
          firstDrawable: firstDrawable,
          lastDrawable: lastDrawable
        } );
      }

      sceneryLog && sceneryLog.Stitch && sceneryLog.pop();
    },

    // Note a block's tentative first drawable and block before (should be flushed later with updateBlockIntervals())
    markBeforeBlock: function( block, firstDrawable ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'marking block first drawable ' + block.toString() + ' with ' + firstDrawable.toString() );

      block.pendingFirstDrawable = firstDrawable;
      this.touchedBlocks.push( block );
    },
    // Note a block's tentative last drawable and block after (should be flushed later with updateBlockIntervals())
    markAfterBlock: function( block, lastDrawable ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'marking block last drawable ' + block.toString() + ' with ' + lastDrawable.toString() );

      block.pendingLastDrawable = lastDrawable;
      this.touchedBlocks.push( block );
    },
    // Flushes markBeforeBlock/markAfterBlock changes to notifyInterval on blocks themselves.
    updateBlockIntervals: function() {
      while ( this.touchedBlocks.length ) {
        var block = this.touchedBlocks.pop();

        if ( block.used ) {
          sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'update interval: ' + block.toString() + ' ' +
                                                                block.pendingFirstDrawable.toString() + ' to ' + block.pendingLastDrawable.toString() );

          block.updateInterval();

          // mark it dirty, since its drawables probably changed?
          //OHTWO TODO: is this necessary? What is this doing?
          this.backbone.markDirtyDrawable( block );

          if ( assertSlow ) {
            this.intervalsNotified.push( {
              block: block,
              firstDrawable: block.pendingFirstDrawable,
              lastDrawable: block.pendingLastDrawable
            } );
          }
        }
        else {
          sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'skipping update interval: ' + block.toString() + ', unused' );
        }
      }
    },

    // Creates a fresh block with the desired renderer and {Drawable} arbitrary drawable included, and adds it to
    // our DOM.
    createBlock: function( renderer, drawable ) {
      var backbone = this.backbone;
      var block;

      if ( Renderer.isCanvas( renderer ) ) {
        block = CanvasBlock.createFromPool( backbone.display, renderer, backbone.transformRootInstance, backbone.backboneInstance );
      }
      else if ( Renderer.isSVG( renderer ) ) {
        //OHTWO TODO: handle filter root separately from the backbone instance?
        block = SVGBlock.createFromPool( backbone.display, renderer, backbone.transformRootInstance, backbone.backboneInstance );
      }
      else if ( Renderer.isDOM( renderer ) ) {
        block = DOMBlock.createFromPool( backbone.display, drawable );
      }
      else if ( Renderer.isWebGL( renderer ) ) {
        block = WebGLBlock.createFromPool( backbone.display, renderer, backbone.transformRootInstance, backbone.backboneInstance );
      }
      else {
        throw new Error( 'unsupported renderer for createBlock: ' + renderer );
      }

      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'created block: ' + block.toString() +
                                                            ' with renderer: ' + renderer +
                                                            ' for drawable: ' + drawable.toString() );

      block.setBlockBackbone( backbone );

      //OHTWO TODO: minor speedup by appending only once its fragment is constructed? or use DocumentFragment?
      backbone.domElement.appendChild( block.domElement );

      // if backbone is a display root, hide all of its content from screen readers
      if ( backbone.isDisplayRoot ) {
        block.domElement.setAttribute( 'aria-hidden', true );
      }

      // mark it dirty for now, so we can check
      backbone.markDirtyDrawable( block );

      if ( assertSlow ) {
        this.createdBlocks.push( {
          block: block,
          renderer: renderer,
          drawable: drawable
        } );
      }

      return block;
    },

    // Immediately appends a block to our blocks array
    appendBlock: function( block ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'appending block: ' + block.toString() );

      this.backbone.blocks.push( block );

      if ( assertSlow ) {
        this.reindexed = false;
      }
    },

    // Immediately removes a block to our blocks array
    removeBlock: function( block ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'removing block: ' + block.toString() );

      // remove the block from our internal list
      var blockIndex = _.indexOf( this.backbone.blocks, block );
      assert && assert( blockIndex >= 0, 'Cannot remove block, not attached: ' + block.toString() );
      this.backbone.blocks.splice( blockIndex, 1 );

      if ( assertSlow ) {
        this.reindexed = false;
      }
    },

    useNoBlocks: function() {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'using no blocks' );

      // i.e. we will not use any blocks
      cleanArray( this.backbone.blocks );
    },

    // Triggers all blocks in the blocks array to have their z-index properties set so that they visually stack
    // correctly.
    reindex: function() {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'reindexing blocks' );

      this.backbone.reindexBlocks();

      if ( assertSlow ) {
        this.reindexed = true;
      }
    },

    // An audit for testing assertions
    auditStitch: function() {
      if ( assertSlow ) {
        var self = this;

        var blocks = self.backbone.blocks;
        var previousBlocks = self.previousBlocks;

        assertSlow( self.initialized, 'We seem to have finished a stitch without proper initialization' );
        assertSlow( self.boundariesRecorded, 'Our stitch API requires recordBackboneBoundaries() to be called before' +
                                                 ' it is finished.' );

        // ensure our indices are up-to-date (reindexed, or did not change)
        assertSlow( self.reindexed || blocks.length === 0 ||
                    // array equality of previousBlocks and blocks
                    ( previousBlocks.length === blocks.length &&
                      _.every( _.zip( previousBlocks, blocks ), function( arr ) {
                        return arr[ 0 ] === arr[ 1 ];
                      } ) ),
          'Did not reindex on a block change where we are left with blocks' );

        // all created blocks had intervals notified
        _.each( self.createdBlocks, function( blockData ) {
          assertSlow( _.some( self.intervalsNotified, function( intervalData ) {
            return blockData.block === intervalData.block;
          } ), 'Created block does not seem to have an interval notified: ' + blockData.block.toString() );
        } );

        // no disposed blocks had intervals notified
        _.each( self.disposedBlocks, function( blockData ) {
          assertSlow( !_.some( self.intervalsNotified, function( intervalData ) {
            return blockData.block === intervalData.block;
          } ), 'Removed block seems to have an interval notified: ' + blockData.block.toString() );
        } );

        // all drawables for disposed blocks have been marked as pending removal (or moved)
        _.each( self.disposedBlocks, function( blockData ) {
          var block = blockData.block;
          _.each( Drawable.oldListToArray( block.firstDrawable, block.lastDrawable ), function( drawable ) {
            assertSlow( _.some( self.pendingRemovals, function( removalData ) {
                return removalData.drawable === drawable;
              } ) || _.some( self.pendingMoves, function( moveData ) {
                return moveData.drawable === drawable;
              } ), 'Drawable ' + drawable.toString() + ' originally listed for disposed block ' + block.toString() +
                   ' does not seem to be marked for pending removal or move!' );
          } );
        } );

        // all drawables for created blocks have been marked as pending addition or moved for our block
        _.each( self.createdBlocks, function( blockData ) {
          var block = blockData.block;
          _.each( Drawable.listToArray( block.pendingFirstDrawable, block.pendingLastDrawable ), function( drawable ) {
            assertSlow( _.some( self.pendingAdditions, function( additionData ) {
                return additionData.drawable === drawable && additionData.block === block;
              } ) || _.some( self.pendingMoves, function( moveData ) {
                return moveData.drawable === drawable && moveData.block === block;
              } ), 'Drawable ' + drawable.toString() + ' now listed for created block ' + block.toString() +
                   ' does not seem to be marked for pending addition or move!' );
          } );
        } );

        // all disposed blocks should have been removed
        _.each( self.disposedBlocks, function( blockData ) {
          var blockIdx = _.indexOf( blocks, blockData.block );
          assertSlow( blockIdx < 0, 'Disposed block ' + blockData.block.toString() + ' still present at index ' + blockIdx );
        } );

        // all created blocks should have been added
        _.each( self.createdBlocks, function( blockData ) {
          var blockIdx = _.indexOf( blocks, blockData.block );
          assertSlow( blockIdx >= 0, 'Created block ' + blockData.block.toString() + ' is not in the blocks array' );
        } );

        // all current blocks should be marked as used
        _.each( blocks, function( block ) {
          assertSlow( block.used, 'All current blocks should be marked as used' );
        } );

        assertSlow( blocks.length - previousBlocks.length === self.createdBlocks.length - self.disposedBlocks.length,
          'The count of unmodified blocks should be constant (equal differences):\n' +
          'created: ' + _.map( self.createdBlocks, function( n ) { return n.block.id; } ).join( ',' ) + '\n' +
          'disposed: ' + _.map( self.disposedBlocks, function( n ) { return n.block.id; } ).join( ',' ) + '\n' +
          'before: ' + _.map( previousBlocks, function( n ) { return n.id; } ).join( ',' ) + '\n' +
          'after: ' + _.map( blocks, function( n ) { return n.id; } ).join( ',' ) );

        assertSlow( this.touchedBlocks.length === 0,
          'If we marked any blocks for changes, we should have called updateBlockIntervals' );

        if ( blocks.length ) {

          assertSlow( self.backbone.previousFirstDrawable !== null &&
                      self.backbone.previousLastDrawable !== null,
            'If we are left with at least one block, we must be tracking at least one drawable' );

          assertSlow( blocks[ 0 ].pendingFirstDrawable === self.backbone.previousFirstDrawable,
            'Our first drawable should match the first drawable of our first block' );

          assertSlow( blocks[ blocks.length - 1 ].pendingLastDrawable === self.backbone.previousLastDrawable,
            'Our last drawable should match the last drawable of our last block' );

          for ( var i = 0; i < blocks.length - 1; i++ ) {
            // [i] and [i+1] are a pair of consecutive blocks
            assertSlow( blocks[ i ].pendingLastDrawable.nextDrawable === blocks[ i + 1 ].pendingFirstDrawable &&
                        blocks[ i ].pendingLastDrawable === blocks[ i + 1 ].pendingFirstDrawable.previousDrawable,
              'Consecutive blocks should have boundary drawables that are also consecutive in the linked list' );
          }
        }
        else {
          assertSlow( self.backbone.previousFirstDrawable === null &&
                      self.backbone.previousLastDrawable === null,
            'If we are left with no blocks, it must mean we are tracking precisely zero drawables' );
        }
      }
    }
  } );

  Stitcher.debugIntervals = function( firstChangeInterval ) {
    if ( sceneryLog && sceneryLog.Stitch ) {
      for ( var debugInterval = firstChangeInterval; debugInterval !== null; debugInterval = debugInterval.nextChangeInterval ) {
        sceneryLog.Stitch( '  interval: ' +
                           ( debugInterval.isEmpty() ? '(empty) ' : '' ) +
                           ( debugInterval.drawableBefore ? debugInterval.drawableBefore.toString() : '-' ) + ' to ' +
                           ( debugInterval.drawableAfter ? debugInterval.drawableAfter.toString() : '-' ) );
      }
    }
  };

  // logs a bunch of information about the old (useCurrent===false) or new (useCurrent===true) drawable linked list.
  Stitcher.debugDrawables = function( firstDrawable, lastDrawable, firstChangeInterval, lastChangeInterval, useCurrent ) {
    if ( sceneryLog && sceneryLog.StitchDrawables ) {
      if ( firstDrawable === null ) {
        sceneryLog.StitchDrawables( 'nothing', 'color: #666;' );
        return;
      }

      var isChanged = firstChangeInterval.drawableBefore === null;
      var currentInterval = firstChangeInterval;

      for ( var drawable = firstDrawable; ; drawable = ( useCurrent ? drawable.nextDrawable : drawable.oldNextDrawable ) ) {
        if ( isChanged && drawable === currentInterval.drawableAfter ) {
          isChanged = false;
          currentInterval = currentInterval.nextChangeInterval;
        }

        var drawableString = drawable.renderer + ' ' + ( ( !useCurrent && drawable.parentDrawable ) ? drawable.parentDrawable.toString() : '' ) + ' ' + drawable.toDetailedString();
        sceneryLog.StitchDrawables( drawableString, isChanged ? ( useCurrent ? 'color: #0a0;' : 'color: #a00;' ) : 'color: #666' );

        if ( !isChanged && currentInterval && currentInterval.drawableBefore === drawable ) {
          isChanged = true;
        }

        if ( drawable === lastDrawable ) {
          break;
        }
      }
    }
  };

  return Stitcher;
} );
