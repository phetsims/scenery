// Copyright 2013-2022, University of Colorado Boulder

/**
 * A DOM drawable (div element) that contains child blocks (and is placed in the main DOM tree when visible). It should
 * use z-index for properly ordering its blocks in the correct stacking order.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import cleanArray from '../../../phet-core/js/cleanArray.js';
import Poolable from '../../../phet-core/js/Poolable.js';
import { Drawable, GreedyStitcher, RebuildStitcher, scenery, Stitcher, Utils } from '../imports.js';

// constants
const useGreedyStitcher = true;

class BackboneDrawable extends Drawable {
  /**
   * @mixes Poolable
   *
   * @param {Display} display
   * @param {Instance} backboneInstance
   * @param {Instance} transformRootInstance
   * @param {number} renderer
   * @param {boolean} isDisplayRoot
   */
  constructor( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot ) {
    super();

    this.initialize( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot );
  }

  /**
   * @public
   *
   * @param {Display} display
   * @param {Instance} backboneInstance
   * @param {Instance} transformRootInstance
   * @param {number} renderer
   * @param {boolean} isDisplayRoot
   */
  initialize( display, backboneInstance, transformRootInstance, renderer, isDisplayRoot ) {
    super.initialize( renderer );

    this.display = display;

    // @public {Instance} - reference to the instance that controls this backbone
    this.backboneInstance = backboneInstance;

    // @public {Instance} - where is the transform root for our generated blocks?
    this.transformRootInstance = transformRootInstance;

    // @private {Instance} - where have filters been applied to up? our responsibility is to apply filters between this
    // and our backboneInstance
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
    this.backboneInstance.relativeVisibleEmitter.addListener( this.backboneVisibilityListener );
    this.updateBackboneVisibility();
    this.visibilityDirty = true;

    this.renderer = renderer;
    this.domElement = isDisplayRoot ? display.domElement : BackboneDrawable.createDivBackbone();
    this.isDisplayRoot = isDisplayRoot;
    this.dirtyDrawables = cleanArray( this.dirtyDrawables );

    // Apply CSS needed for future CSS transforms to work properly.
    Utils.prepareForTransform( this.domElement );

    // Ff we need to, watch nodes below us (and including us) and apply their filters (opacity/visibility/clip) to the
    // backbone. Order will be important, since we'll visit them in the order of filter application
    this.watchedFilterNodes = cleanArray( this.watchedFilterNodes );

    // @private {boolean}
    this.filterDirty = true;

    // @private {boolean}
    this.clipDirty = true;

    this.filterDirtyListener = this.filterDirtyListener || this.onFilterDirty.bind( this );
    this.clipDirtyListener = this.clipDirtyListener || this.onClipDirty.bind( this );
    if ( this.willApplyFilters ) {
      assert && assert( this.filterRootAncestorInstance.trail.nodes.length < this.backboneInstance.trail.nodes.length,
        'Our backboneInstance should be deeper if we are applying filters' );

      // walk through to see which instances we'll need to watch for filter changes
      // NOTE: order is important, so that the filters are applied in the correct order!
      for ( let instance = this.backboneInstance; instance !== this.filterRootAncestorInstance; instance = instance.parent ) {
        const node = instance.node;

        this.watchedFilterNodes.push( node );
        node.filterChangeEmitter.addListener( this.filterDirtyListener );
        node.clipAreaProperty.lazyLink( this.clipDirtyListener );
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

    sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( `initialized ${this.toString()}` );
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( `dispose ${this.toString()}` );
    sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.push();


    while ( this.watchedFilterNodes.length ) {
      const node = this.watchedFilterNodes.pop();

      node.filterChangeEmitter.removeListener( this.filterDirtyListener );
      node.clipAreaProperty.unlink( this.clipDirtyListener );
    }

    this.backboneInstance.relativeVisibleEmitter.removeListener( this.backboneVisibilityListener );

    // if we need to remove drawables from the blocks, do so
    if ( !this.removedDrawables ) {
      for ( let d = this.previousFirstDrawable; d !== null; d = d.nextDrawable ) {
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

    super.dispose();

    sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.pop();
  }

  /**
   * Dispose all of the blocks while clearing our references to them
   * @public
   */
  markBlocksForDisposal() {
    while ( this.blocks.length ) {
      const block = this.blocks.pop();
      sceneryLog && sceneryLog.BackboneDrawable && sceneryLog.BackboneDrawable( `${this.toString()} removing block: ${block.toString()}` );
      //TODO: PERFORMANCE: does this cause reflows / style calculation
      if ( block.domElement.parentNode === this.domElement ) {
        // guarded, since we may have a (new) child drawable add it before we can remove it
        this.domElement.removeChild( block.domElement );
      }
      block.markForDisposal( this.display );
    }
  }

  /**
   * @private
   */
  updateBackboneVisibility() {
    this.visible = this.backboneInstance.relativeVisible;

    if ( !this.visibilityDirty ) {
      this.visibilityDirty = true;
      this.markDirty();
    }
  }

  /**
   * Marks this backbone for disposal.
   * @public
   * @override
   *
   * NOTE: Should be called during syncTree
   *
   * @param {Display} display
   */
  markForDisposal( display ) {
    for ( let d = this.previousFirstDrawable; d !== null; d = d.oldNextDrawable ) {
      d.notePendingRemoval( this.display );
      if ( d === this.previousLastDrawable ) { break; }
    }
    this.removedDrawables = true;

    // super call
    super.markForDisposal( display );
  }

  /**
   * Marks a drawable as dirty.
   * @public
   *
   * @param {Drawable} drawable
   */
  markDirtyDrawable( drawable ) {
    if ( assert ) {
      // Catch infinite loops
      this.display.ensureNotPainting();
    }

    this.dirtyDrawables.push( drawable );
    this.markDirty();
  }

  /**
   * Marks our transform as dirty.
   * @public
   */
  markTransformDirty() {
    assert && assert( this.willApplyTransform, 'Sanity check for willApplyTransform' );

    // relative matrix on backbone instance should be up to date, since we added the compute flags
    Utils.applyPreparedTransform( this.backboneInstance.relativeTransform.matrix, this.domElement );
  }

  /**
   * Marks our opacity as dirty.
   * @private
   */
  onFilterDirty() {
    if ( !this.filterDirty ) {
      this.filterDirty = true;
      this.markDirty();
    }
  }

  /**
   * Marks our clip as dirty.
   * @private
   */
  onClipDirty() {
    if ( !this.clipDirty ) {
      this.clipDirty = true;
      this.markDirty();
    }
  }

  /**
   * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
   * @public
   * @override
   *
   * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
   *                      be done).
   */
  update() {
    // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
    if ( !super.update() ) {
      return false;
    }

    while ( this.dirtyDrawables.length ) {
      this.dirtyDrawables.pop().update();
    }

    if ( this.filterDirty ) {
      this.filterDirty = false;

      let filterString = '';

      const len = this.watchedFilterNodes.length;
      for ( let i = 0; i < len; i++ ) {
        const node = this.watchedFilterNodes[ i ];
        const opacity = node.getEffectiveOpacity();

        for ( let j = 0; j < node._filters.length; j++ ) {
          filterString += `${filterString ? ' ' : ''}${node._filters[ j ].getCSSFilterString()}`;
        }

        // Apply opacity after other effects
        if ( opacity !== 1 ) {
          filterString += `${filterString ? ' ' : ''}opacity(${toSVGNumber( opacity )})`;
        }
      }

      this.domElement.style.filter = filterString;
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
  }

  /**
   * Returns the combined visibility of nodes "above us" that will need to be taken into account for displaying this
   * backbone.
   * @public
   *
   * @returns {boolean}
   */
  getFilterVisibility() {
    const len = this.watchedFilterNodes.length;
    for ( let i = 0; i < len; i++ ) {
      if ( !this.watchedFilterNodes[ i ].isVisible() ) {
        return false;
      }
    }

    return true;
  }

  /**
   * Returns the combined clipArea (string???) for nodes "above us".
   * @public
   *
   * @returns {string}
   */
  getFilterClip() {
    const clip = '';

    //OHTWO TODO: proper clipping support
    // var len = this.watchedFilterNodes.length;
    // for ( var i = 0; i < len; i++ ) {
    //   if ( this.watchedFilterNodes[i].clipArea ) {
    //     throw new Error( 'clip-path for backbones unimplemented, and with questionable browser support!' );
    //   }
    // }

    return clip;
  }

  /**
   * Ensures that z-indices are strictly increasing, while trying to minimize the number of times we must change it
   * @public
   */
  reindexBlocks() {
    // full-pass change for zindex.
    let zIndex = 0; // don't start below 1 (we ensure > in loop)
    for ( let k = 0; k < this.blocks.length; k++ ) {
      const block = this.blocks[ k ];
      if ( block.zIndex <= zIndex ) {
        const newIndex = ( k + 1 < this.blocks.length && this.blocks[ k + 1 ].zIndex - 1 > zIndex ) ?
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
  }

  /**
   * Stitches multiple change intervals.
   * @public
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   * @param {ChangeInterval} firstChangeInterval
   * @param {ChangeInterval} lastChangeInterval
   */
  stitch( firstDrawable, lastDrawable, firstChangeInterval, lastChangeInterval ) {
    // no stitch necessary if there are no change intervals
    if ( firstChangeInterval === null || lastChangeInterval === null ) {
      assert && assert( firstChangeInterval === null );
      assert && assert( lastChangeInterval === null );
      return;
    }

    assert && assert( lastChangeInterval.nextChangeInterval === null, 'This allows us to have less checks in the loop' );

    if ( sceneryLog && sceneryLog.Stitch ) {
      sceneryLog.Stitch( `Stitch intervals before constricting: ${this.toString()}` );
      sceneryLog.push();
      Stitcher.debugIntervals( firstChangeInterval );
      sceneryLog.pop();
    }

    // Make the intervals as small as possible by skipping areas without changes, and collapse the interval
    // linked list
    let lastNonemptyInterval = null;
    let interval = firstChangeInterval;
    let intervalsChanged = false;
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
      sceneryLog.Stitch( `Stitch intervals after constricting: ${this.toString()}` );
      sceneryLog.push();
      Stitcher.debugIntervals( firstChangeInterval );
      sceneryLog.pop();
    }

    if ( sceneryLog && scenery.isLoggingPerformance() ) {
      this.display.perfStitchCount++;

      let dInterval = firstChangeInterval;

      while ( dInterval ) {
        this.display.perfIntervalCount++;

        this.display.perfDrawableOldIntervalCount += dInterval.getOldInternalDrawableCount( this.previousFirstDrawable, this.previousLastDrawable );
        this.display.perfDrawableNewIntervalCount += dInterval.getNewInternalDrawableCount( firstDrawable, lastDrawable );

        dInterval = dInterval.nextChangeInterval;
      }
    }

    this.stitcher.stitch( this, firstDrawable, lastDrawable, this.previousFirstDrawable, this.previousLastDrawable, firstChangeInterval, lastChangeInterval );
  }

  /**
   * Runs checks on the drawable, based on certain flags.
   * @public
   * @override
   *
   * @param {boolean} allowPendingBlock
   * @param {boolean} allowPendingList
   * @param {boolean} allowDirty
   */
  audit( allowPendingBlock, allowPendingList, allowDirty ) {
    if ( assertSlow ) {
      super.audit( allowPendingBlock, allowPendingList, allowDirty );

      assertSlow && assertSlow( this.backboneInstance.isBackbone, 'We should reference an instance that requires a backbone' );
      assertSlow && assertSlow( this.transformRootInstance.isTransformed, 'Transform root should be transformed' );

      for ( let i = 0; i < this.blocks.length; i++ ) {
        this.blocks[ i ].audit( allowPendingBlock, allowPendingList, allowDirty );
      }
    }
  }

  /**
   * Creates a base DOM element for a backbone.
   * @public
   *
   * @returns {HTMLDivElement}
   */
  static createDivBackbone() {
    const div = document.createElement( 'div' );
    div.style.position = 'absolute';
    div.style.left = '0';
    div.style.top = '0';
    div.style.width = '0';
    div.style.height = '0';
    return div;
  }

  /**
   * Given an external element, we apply the necessary style to make it compatible as a backbone DOM element.
   * @public
   *
   * @param {HTMLElement} element
   * @returns {HTMLElement} - For chaining
   */
  static repurposeBackboneContainer( element ) {
    if ( element.style.position !== 'relative' || element.style.position !== 'absolute' ) {
      element.style.position = 'relative';
    }
    element.style.left = '0';
    element.style.top = '0';
    return element;
  }
}

scenery.register( 'BackboneDrawable', BackboneDrawable );

Poolable.mixInto( BackboneDrawable );

export default BackboneDrawable;