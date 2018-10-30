// Copyright 2013-2016, University of Colorado Boulder

/**
 * A DOM drawable (div element) that contains child blocks (and is placed in the main DOM tree when visible). It should
 * use z-index for properly ordering its blocks in the correct stacking order.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  // modules
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  var GreedyStitcher = require( 'SCENERY/display/GreedyStitcher' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var RebuildStitcher = require( 'SCENERY/display/RebuildStitcher' );
  var scenery = require( 'SCENERY/scenery' );
  var Stitcher = require( 'SCENERY/display/Stitcher' );
  var Util = require( 'SCENERY/util/Util' );

  // constants
  var useGreedyStitcher = true;

  /**
   * @constructor
   * @mixes Poolable
   *
   * @param {Display} display
   * @param {Instance} backboneInstance
   * @param {Instance} transformRootInstance
   * @param {number} renderer
   * @param {boolean} isDisplayRoot
   */
  function BackboneDrawable( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot ) {
    this.initialize( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot );
  }

  scenery.register( 'BackboneDrawable', BackboneDrawable );

  inherit( Drawable, BackboneDrawable, {

    /**
     * @param {Display} display
     * @param {Instance} backboneInstance
     * @param {Instance} transformRootInstance
     * @param {number} renderer
     * @param {boolean} isDisplayRoot
     * @returns {BackboneDrawable} - Returns 'this' reference, for chaining
     */
    initialize: function( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot ) {
      Drawable.call( this, renderer );

      this.display = display;

      this.forceAcceleration = false;

      // reference to the instance that controls this backbone
      this.backboneInstance = backboneInstance;

      // where is the transform root for our generated blocks?
      this.transformRootInstance = transformRootInstance;

      // where have filters been applied to up? our responsibility is to apply filters between this and our backboneInstance
      this.filterRootAncestorInstance = backboneInstance.parent ? backboneInstance.parent.getFilterRootInstance() : backboneInstance;

      // where have transforms been applied up to? our responsibility is to apply transforms between this and our backboneInstance
      this.transformRootAncestorInstance = backboneInstance.parent ? backboneInstance.parent.getTransformRootInstance() : backboneInstance;

      this.willApplyTransform = this.transformRootAncestorInstance !== this.transformRootInstance;
      this.willApplyFilters = this.filterRootAncestorInstance !== this.backboneInstance;

      this.transformListener = this.transformListener || this.markTransformDirty.bind( this );
      if ( this.willApplyTransform ) {
        this.backboneInstance.relativeTransform.addListener( this.transformListener ); // when our relative transform changes, notify us in the pre-repaint phase
        this.backboneInstance.relativeTransform.addPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated
      }

      this.backboneVisibilityListener = this.backboneVisibilityListener || this.updateBackboneVisibility.bind( this );
      this.backboneInstance.onStatic( 'relativeVisibility', this.backboneVisibilityListener );
      this.updateBackboneVisibility();
      this.visibilityDirty = true;

      this.renderer = renderer;
      this.domElement = isDisplayRoot ? display._domElement : BackboneDrawable.createDivBackbone();
      this.isDisplayRoot = isDisplayRoot;
      this.dirtyDrawables = cleanArray( this.dirtyDrawables );

      // Apply CSS needed for future CSS transforms to work properly.
      Util.prepareForTransform( this.domElement, this.forceAcceleration );

      // if we need to, watch nodes below us (and including us) and apply their filters (opacity/visibility/clip) to the backbone.
      this.watchedFilterNodes = cleanArray( this.watchedFilterNodes );
      this.opacityDirty = true;
      this.clipDirty = true;
      this.opacityDirtyListener = this.opacityDirtyListener || this.markOpacityDirty.bind( this );
      this.clipDirtyListener = this.clipDirtyListener || this.markClipDirty.bind( this );
      if ( this.willApplyFilters ) {
        assert && assert( this.filterRootAncestorInstance.trail.nodes.length < this.backboneInstance.trail.nodes.length,
          'Our backboneInstance should be deeper if we are applying filters' );

        // walk through to see which instances we'll need to watch for filter changes
        for ( var instance = this.backboneInstance; instance !== this.filterRootAncestorInstance; instance = instance.parent ) {
          var node = instance.node;

          this.watchedFilterNodes.push( node );
          node.onStatic( 'opacity', this.opacityDirtyListener );
          node.onStatic( 'clip', this.clipDirtyListener );
        }
      }

      this.lastZIndex = 0; // our last zIndex is stored, so that overlays can be added easily

      this.blocks = this.blocks || []; // we are responsible for their disposal

      // the first/last drawables for the last the this backbone was stitched
      this.previousFirstDrawable = null;
      this.previousLastDrawable = null;

      // We track whether our drawables were marked for removal (in which case, they should all be removed by the time we dispose).
      // If removedDrawables = false during disposal, it means we need to remove the drawables manually (this should only happen if an instance tree is removed)
      this.removedDrawables = false;

      this.stitcher = this.stitcher || ( useGreedyStitcher ? new GreedyStitcher() : new RebuildStitcher() );

      sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( 'initialized ' + this.toString() );

      return this; // chaining
    },

    dispose: function() {
      sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( 'dispose ' + this.toString() );
      sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.push();


      while ( this.watchedFilterNodes.length ) {
        var node = this.watchedFilterNodes.pop();

        node.offStatic( 'opacity', this.opacityDirtyListener );
        node.offStatic( 'clip', this.clipDirtyListener );
      }

      this.backboneInstance.offStatic( 'relativeVisibility', this.backboneVisibilityListener );

      // if we need to remove drawables from the blocks, do so
      if ( !this.removedDrawables ) {
        for ( var d = this.previousFirstDrawable; d !== null; d = d.nextDrawable ) {
          d.parentDrawable.removeDrawable( d );
          if ( d === this.previousLastDrawable ) { break; }
        }
      }

      this.markBlocksForDisposal();

      if ( this.willApplyTransform ) {
        this.backboneInstance.relativeTransform.removeListener( this.transformListener );
        this.backboneInstance.relativeTransform.removePrecompute();
      }

      this.backboneInstance = null;
      this.transformRootInstance = null;
      this.filterRootAncestorInstance = null;
      this.transformRootAncestorInstance = null;
      cleanArray( this.dirtyDrawables );
      cleanArray( this.watchedFilterNodes );

      this.previousFirstDrawable = null;
      this.previousLastDrawable = null;

      Drawable.prototype.dispose.call( this );

      sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.pop();
    },

    // dispose all of the blocks while clearing our references to them
    markBlocksForDisposal: function() {
      while ( this.blocks.length ) {
        var block = this.blocks.pop();
        sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( this.toString() + ' removing block: ' + block.toString() );
        //TODO: PERFORMANCE: does this cause reflows / style calculation
        if ( block.domElement.parentNode === this.domElement ) {
          // guarded, since we may have a (new) child drawable add it before we can remove it
          this.domElement.removeChild( block.domElement );
        }
        block.markForDisposal( this.display );
      }
    },

    updateBackboneVisibility: function() {
      this.visible = this.backboneInstance.relativeVisible;

      if ( !this.visibilityDirty ) {
        this.visibilityDirty = true;
        this.markDirty();
      }
    },

    // should be called during syncTree
    markForDisposal: function( display ) {
      for ( var d = this.previousFirstDrawable; d !== null; d = d.oldNextDrawable ) {
        d.notePendingRemoval( this.display );
        if ( d === this.previousLastDrawable ) { break; }
      }
      this.removedDrawables = true;

      // super call
      Drawable.prototype.markForDisposal.call( this, display );
    },

    markDirtyDrawable: function( drawable ) {
      if ( assert ) {
        // Catch infinite loops
        this.display.ensureNotPainting();
      }

      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },

    markTransformDirty: function() {
      assert && assert( this.willApplyTransform, 'Sanity check for willApplyTransform' );

      // relative matrix on backbone instance should be up to date, since we added the compute flags
      scenery.Util.applyPreparedTransform( this.backboneInstance.relativeTransform.matrix, this.domElement, this.forceAcceleration );
    },

    markOpacityDirty: function() {
      if ( !this.opacityDirty ) {
        this.opacityDirty = true;
        this.markDirty();
      }
    },

    markClipDirty: function() {
      if ( !this.clipDirty ) {
        this.clipDirty = true;
        this.markDirty();
      }
    },

    /**
     * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
     * @public
     * @override
     *
     * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
     *                      be done).
     */
    update: function() {
      // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
      if ( !Drawable.prototype.update.call( this ) ) {
        return false;
      }

      while ( this.dirtyDrawables.length ) {
        this.dirtyDrawables.pop().update();
      }

      if ( this.opacityDirty ) {
        this.opacityDirty = false;

        var filterOpacity = this.willApplyFilters ? this.getFilterOpacity() : 1;
        this.domElement.style.opacity = ( filterOpacity !== 1 ) ? filterOpacity : '';
      }

      if ( this.visibilityDirty ) {
        this.visibilityDirty = false;

        this.domElement.style.display = this.visible ? '' : 'none';
      }

      if ( this.clipDirty ) {
        this.clipDirty = false;

        // var clip = this.willApplyFilters ? this.getFilterClip() : '';

        //OHTWO TODO: CSS clip-path/mask support here. see http://www.html5rocks.com/en/tutorials/masking/adobe/
        // this.domElement.style.clipPath = clip; // yikes! temporary, since we already threw something?
      }

      return true;
    },

    getFilterOpacity: function() {
      var opacity = 1;

      var len = this.watchedFilterNodes.length;
      for ( var i = 0; i < len; i++ ) {
        opacity *= this.watchedFilterNodes[ i ].getOpacity();
      }

      return opacity;
    },

    getFilterVisibility: function() {
      var len = this.watchedFilterNodes.length;
      for ( var i = 0; i < len; i++ ) {
        if ( !this.watchedFilterNodes[ i ].isVisible() ) {
          return false;
        }
      }

      return true;
    },

    getFilterClip: function() {
      var clip = '';

      //OHTWO TODO: proper clipping support
      // var len = this.watchedFilterNodes.length;
      // for ( var i = 0; i < len; i++ ) {
      //   if ( this.watchedFilterNodes[i].clipArea ) {
      //     throw new Error( 'clip-path for backbones unimplemented, and with questionable browser support!' );
      //   }
      // }

      return clip;
    },

    // ensures that z-indices are strictly increasing, while trying to minimize the number of times we must change it
    reindexBlocks: function() {
      // full-pass change for zindex.
      var zIndex = 0; // don't start below 1 (we ensure > in loop)
      for ( var k = 0; k < this.blocks.length; k++ ) {
        var block = this.blocks[ k ];
        if ( block.zIndex <= zIndex ) {
          var newIndex = ( k + 1 < this.blocks.length && this.blocks[ k + 1 ].zIndex - 1 > zIndex ) ?
                         Math.ceil( ( zIndex + this.blocks[ k + 1 ].zIndex ) / 2 ) :
                         zIndex + 20;

          // NOTE: this should give it its own stacking index (which is what we want)
          block.domElement.style.zIndex = block.zIndex = newIndex;
        }
        zIndex = block.zIndex;

        if ( assert ) {
          assert( this.blocks[ k ].zIndex % 1 === 0, 'z-indices should be integers' );
          assert( this.blocks[ k ].zIndex > 0, 'z-indices should be greater than zero for our needs (see spec)' );
          if ( k > 0 ) {
            assert( this.blocks[ k - 1 ].zIndex < this.blocks[ k ].zIndex, 'z-indices should be strictly increasing' );
          }
        }
      }

      // sanity check
      this.lastZIndex = zIndex + 1;
    },

    stitch: function( firstDrawable, lastDrawable, firstChangeInterval, lastChangeInterval ) {
      // no stitch necessary if there are no change intervals
      if ( firstChangeInterval === null || lastChangeInterval === null ) {
        assert && assert( firstChangeInterval === null );
        assert && assert( lastChangeInterval === null );
        return;
      }

      assert && assert( lastChangeInterval.nextChangeInterval === null, 'This allows us to have less checks in the loop' );

      if ( sceneryLog && sceneryLog.Stitch ) {
        sceneryLog.Stitch( 'Stitch intervals before constricting: ' + this.toString() );
        sceneryLog.push();
        Stitcher.debugIntervals( firstChangeInterval );
        sceneryLog.pop();
      }

      // Make the intervals as small as possible by skipping areas without changes, and collapse the interval
      // linked list
      var lastNonemptyInterval = null;
      var interval = firstChangeInterval;
      var intervalsChanged = false;
      while ( interval ) {
        intervalsChanged = interval.constrict() || intervalsChanged;

        if ( interval.isEmpty() ) {
          assert && assert( intervalsChanged );

          if ( lastNonemptyInterval ) {
            // skip it, hook the correct reference
            lastNonemptyInterval.nextChangeInterval = interval.nextChangeInterval;
          }
        }
        else {
          // our first non-empty interval will be our new firstChangeInterval
          if ( !lastNonemptyInterval ) {
            firstChangeInterval = interval;
          }
          lastNonemptyInterval = interval;
        }
        interval = interval.nextChangeInterval;
      }

      if ( !lastNonemptyInterval ) {
        // eek, no nonempty change intervals. do nothing (good to catch here, but ideally there shouldn't be change
        // intervals that all collapse).
        return;
      }

      lastChangeInterval = lastNonemptyInterval;
      lastChangeInterval.nextChangeInterval = null;

      if ( sceneryLog && sceneryLog.Stitch && intervalsChanged ) {
        sceneryLog.Stitch( 'Stitch intervals after constricting: ' + this.toString() );
        sceneryLog.push();
        Stitcher.debugIntervals( firstChangeInterval );
        sceneryLog.pop();
      }

      if ( sceneryLog && scenery.isLoggingPerformance() ) {
        this.display.perfStitchCount++;

        var dInterval = firstChangeInterval;

        while ( dInterval ) {
          this.display.perfIntervalCount++;

          this.display.perfDrawableOldIntervalCount += dInterval.getOldInternalDrawableCount( this.previousFirstDrawable, this.previousLastDrawable );
          this.display.perfDrawableNewIntervalCount += dInterval.getNewInternalDrawableCount( firstDrawable, lastDrawable );

          dInterval = dInterval.nextChangeInterval;
        }
      }

      this.stitcher.stitch( this, firstDrawable, lastDrawable, this.previousFirstDrawable, this.previousLastDrawable, firstChangeInterval, lastChangeInterval );
    },

    audit: function( allowPendingBlock, allowPendingList, allowDirty ) {
      if ( assertSlow ) {
        Drawable.prototype.audit.call( this, allowPendingBlock, allowPendingList, allowDirty );

        assertSlow && assertSlow( this.backboneInstance.isBackbone, 'We should reference an instance that requires a backbone' );
        assertSlow && assertSlow( this.transformRootInstance.isTransformed, 'Transform root should be transformed' );

        for ( var i = 0; i < this.blocks.length; i++ ) {
          this.blocks[ i ].audit( allowPendingBlock, allowPendingList, allowDirty );
        }
      }
    }
  } );

  BackboneDrawable.createDivBackbone = function() {
    var div = document.createElement( 'div' );
    div.style.position = 'absolute';
    div.style.left = '0';
    div.style.top = '0';
    div.style.width = '0';
    div.style.height = '0';
    return div;
  };

  BackboneDrawable.repurposeBackboneContainer = function( element ) {
    if ( element.style.position !== 'relative' || element.style.position !== 'absolute' ) {
      element.style.position = 'relative';
    }
    element.style.left = '0';
    element.style.top = '0';
    return element;
  };

  Poolable.mixInto( BackboneDrawable, {
    initialize: BackboneDrawable.prototype.initialize
  } );

  return BackboneDrawable;
} );
