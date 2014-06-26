// Copyright 2002-2014, University of Colorado

/**
 * Abstract base type (and API) for stitching implementations.
 *
 * Assumes the same object instance will be reused.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var CanvasBlock = require( 'SCENERY/display/CanvasBlock' );
  var SVGBlock = require( 'SCENERY/display/SVGBlock' );
  var DOMBlock = require( 'SCENERY/display/DOMBlock' );
  
  scenery.Stitcher = function Stitcher( display, renderer ) {
    throw new Error( 'We are too abstract for that! ');
  };
  var Stitcher = scenery.Stitcher;
  
  inherit( Object, Stitcher, {
    constructor: scenery.Stitcher,
    
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
    
    notePendingAddition: function( drawable, block ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'pending add: ' + drawable.toString() + ' to ' + block.toString() );
      
      drawable.notePendingAddition( this.backbone.display, block, this.backbone );
      
      if ( assertSlow ) {
        this.pendingAdditions.push( {
          drawable: drawable,
          block: block
        } );
      }
    },
    
    notePendingMove: function( drawable, block ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'pending move: ' + drawable.toString() + ' to ' + block.toString() );
      
      drawable.notePendingMove( this.backbone.display, block );
      
      if ( assertSlow ) {
        this.pendingMoves.push( {
          drawable: drawable,
          block: block
        } );
      }
    },
    
    notePendingRemoval: function( drawable ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'pending remove: ' + drawable.toString() );
      
      drawable.notePendingRemoval( this.backbone.display );
      
      if ( assertSlow ) {
        this.pendingRemovals.push( {
          drawable: drawable
        } );
      }
    },
    
    markBlockForDisposal: function( block ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'block for disposal: ' + block.toString() );
      
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
    },
    
    removeAllBlocks: function() {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'marking all blocks for disposal (count ' + this.backbone.blocks.length + ')' );
      while ( this.backbone.blocks.length ) {
        var block = this.backbone.blocks[0];
        
        this.removeBlock( block );
        this.markBlockForDisposal( block );
      }
    },
    
    notifyInterval: function( block, firstDrawable, lastDrawable ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'notify interval: ' + block.toString() + ' ' +
                                                            firstDrawable.toString() + ' to ' + lastDrawable.toString() );
      
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
    },
    
    // notifyInterval alternatives, so changes can be collected before notifying:
    markBeforeBlock: function( block, firstDrawable ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'marking block first drawable ' + block.toString() + ' with ' + firstDrawable.toString() );
      
      block.pendingFirstDrawable = firstDrawable;
      this.touchedBlocks.push( block );
    },
    markAfterBlock: function( block, lastDrawable ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'marking block last drawable ' + block.toString() + ' with ' + lastDrawable.toString() );
      
      block.pendingLastDrawable = lastDrawable;
      this.touchedBlocks.push( block );
    },
    updateBlockIntervals: function() {
      while ( this.touchedBlocks.length ) {
        var block = this.touchedBlocks.pop();
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
    },
    
    createBlock: function( renderer, drawable ) {
      var backbone = this.backbone;
      var block;
      
      if ( Renderer.isCanvas( renderer ) ) {
        block = CanvasBlock.createFromPool( backbone.display, renderer, backbone.transformRootInstance, backbone.backboneInstance );
      } else if ( Renderer.isSVG( renderer ) ) {
        //OHTWO TODO: handle filter root separately from the backbone instance?
        block = SVGBlock.createFromPool( backbone.display, renderer, backbone.transformRootInstance, backbone.backboneInstance );
      } else if ( Renderer.isDOM( renderer ) ) {
        block = DOMBlock.createFromPool( backbone.display, drawable );
      } else {
        throw new Error( 'unsupported renderer for createBlock: ' + renderer );
      }
      
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'created block: ' + block.toString() +
                                                            ' with renderer: ' + renderer +
                                                            ' for drawable: ' + drawable.toString() );
      
      block.setBlockBackbone( backbone );
      
      //OHTWO TODO: minor speedup by appending only once its fragment is constructed? or use DocumentFragment?
      backbone.domElement.appendChild( block.domElement );
      
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
    
    appendBlock: function( block ) {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'appending block: ' + block.toString() );
      
      this.backbone.blocks.push( block );
      
      if ( assertSlow ) {
        this.reindexed = false;
      }
    },
    
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
    
    reindex: function() {
      sceneryLog && sceneryLog.Stitch && sceneryLog.Stitch( 'reindexing blocks' );
      
      this.backbone.reindexBlocks();
      
      if ( assertSlow ) {
        this.reindexed = true;
      }
    },
    
    auditStitch: function() {
      if ( assertSlow ) {
        var stitcher = this;
        
        var blocks = stitcher.backbone.blocks;
        var previousBlocks = stitcher.previousBlocks;
        
        assertSlow( stitcher.initialized, 'We seem to have finished a stitch without proper initialization' );
        assertSlow( stitcher.boundariesRecorded, 'Our stitch API requires recordBackboneBoundaries() to be called before' +
                                                 ' it is finished.' );
        
        // ensure our indices are up-to-date (reindexed, or didn't change)
        assertSlow( stitcher.reindexed ||
                    // array equality of previousBlocks and blocks
                    ( previousBlocks.length === blocks.length &&
                      _.every( _.zip( previousBlocks, blocks ), function( arr ) {
                        return arr[0] === arr[1];
                      } ) ),
                    'Did not reindex on a block change' );
        
        // all created blocks had intervals notified
        _.each( stitcher.createdBlocks, function( blockData ) {
          assertSlow( _.some( stitcher.intervalsNotified, function( intervalData ) {
            return blockData.block === intervalData.block;
          } ), 'Created block does not seem to have an interval notified: ' + blockData.block.toString() );
        } );
        
        // no disposed blocks had intervals notified
        _.each( stitcher.disposedBlocks, function( blockData ) {
          assertSlow( !_.some( stitcher.intervalsNotified, function( intervalData ) {
            return blockData.block === intervalData.block;
          } ), 'Removed block seems to have an interval notified: ' + blockData.block.toString() );
        } );
        
        // all drawables for disposed blocks have been marked as pending removal (or moved)
        _.each( stitcher.disposedBlocks, function( blockData ) {
          var block = blockData.block;
          _.each( Drawable.oldListToArray( block.firstDrawable, block.lastDrawable ), function( drawable ) {
            assertSlow( _.some( stitcher.pendingRemovals, function( removalData ) {
              return removalData.drawable === drawable;
            } ) || _.some( stitcher.pendingMoves, function( moveData ) {
              return moveData.drawable === drawable;
            } ), 'Drawable ' + drawable.toString() + ' originally listed for disposed block ' + block.toString() +
                 ' does not seem to be marked for pending removal!' );
          } );
        } );
        
        // all drawables for created blocks have been marked as pending addition or moved for our block
        _.each( stitcher.createdBlocks, function( blockData ) {
          var block = blockData.block;
          _.each( Drawable.listToArray( block.pendingFirstDrawable, block.pendingLastDrawable ), function( drawable ) {
            assertSlow( _.some( stitcher.pendingAdditions, function( additionData ) {
              return additionData.drawable === drawable && additionData.block === block;
            } ) || _.some( stitcher.pendingMoves, function( moveData ) {
              return moveData.drawable === drawable && moveData.block === block;
            } ), 'Drawable ' + drawable.toString() + ' now listed for created block ' + block.toString() +
                 ' does not seem to be marked for pending addition or move!' );
          } );
        } );
        
        // all disposed blocks should have been removed
        _.each( stitcher.disposedBlocks, function( blockData ) {
          var blockIdx = _.indexOf( blocks, blockData.block );
          assertSlow( blockIdx < 0, 'Disposed block ' + blockData.block.toString() + ' still present at index ' + blockIdx );
        } );
        
        // all created blocks should have been added
        _.each( stitcher.createdBlocks, function( blockData ) {
          var blockIdx = _.indexOf( blocks, blockData.block );
          assertSlow( blockIdx >= 0, 'Created block ' + blockData.block.toString() + ' is not in the blocks array' );
        } );
        
        assertSlow( blocks.length - previousBlocks.length === stitcher.createdBlocks.length - stitcher.disposedBlocks.length,
                    'The count of unmodified blocks should be constant (equal differences)' );
        
        assertSlow( this.touchedBlocks.length === 0,
                    'If we marked any blocks for changes, we should have called updateBlockIntervals' );
        
        if ( blocks.length ) {
          
          assertSlow( stitcher.backbone.previousFirstDrawable !== null &&
                      stitcher.backbone.previousLastDrawable !== null,
                      'If we are left with at least one block, we must be tracking at least one drawable' );
          
          assertSlow( blocks[0].pendingFirstDrawable === stitcher.backbone.previousFirstDrawable,
                      'Our first drawable should match the first drawable of our first block' );
          
          assertSlow( blocks[blocks.length-1].pendingLastDrawable === stitcher.backbone.previousLastDrawable,
                      'Our last drawable should match the last drawable of our last block' );
          
          for ( var i = 0; i < blocks.length - 1; i++ ) {
            // [i] and [i+1] are a pair of consecutive blocks
            assertSlow( blocks[i].pendingLastDrawable.nextDrawable === blocks[i+1].pendingFirstDrawable &&
                        blocks[i].pendingLastDrawable === blocks[i+1].pendingFirstDrawable.previousDrawable,
                        'Consecutive blocks should have boundary drawables that are also consecutive in the linked list' );
          }
        } else {
          assertSlow( stitcher.backbone.previousFirstDrawable === null &&
                      stitcher.backbone.previousLastDrawable === null,
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
  
  Stitcher.debugDrawables = function( firstDrawable, lastDrawable, firstChangeInterval, lastChangeInterval, useCurrent ) {
    if ( sceneryLog && sceneryLog.StitchDrawables ) {
      if ( firstDrawable === null ) {
        sceneryLog.StitchDrawables( 'nothing', 'color: #666;' );
        return;
      }
      
      var isChanged = firstChangeInterval.drawableBefore === null;
      var currentInterval = firstChangeInterval;
      
      for ( var drawable = firstDrawable;; drawable = ( useCurrent ? drawable.nextDrawable : drawable.oldNextDrawable ) ) {
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
