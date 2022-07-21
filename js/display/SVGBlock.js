// Copyright 2013-2022, University of Colorado Boulder

/**
 * Handles a visual SVG layer of drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import cleanArray from '../../../phet-core/js/cleanArray.js';
import Poolable from '../../../phet-core/js/Poolable.js';
import { CountMap, FittedBlock, scenery, SVGGroup, svgns, Utils } from '../imports.js';

class SVGBlock extends FittedBlock {
  /**
   * @mixes Poolable
   *
   * @param {Display} display - the scenery Display this SVGBlock will appear in
   * @param {number} renderer - the bitmask for the renderer, see Renderer.js
   * @param {Instance} transformRootInstance - TODO: Documentation
   * @param {Instance} filterRootInstance - TODO: Documentation
   */
  constructor( display, renderer, transformRootInstance, filterRootInstance ) {
    super();

    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  }

  /**
   * @public
   *
   * @param {Display} display - the scenery Display this SVGBlock will appear in
   * @param {number} renderer - the bitmask for the renderer, see Renderer.js
   * @param {Instance} transformRootInstance - TODO: Documentation
   * @param {Instance} filterRootInstance - TODO: Documentation
   * @returns {FittedBlock}
   */
  initialize( display, renderer, transformRootInstance, filterRootInstance ) {
    super.initialize( display, renderer, transformRootInstance, FittedBlock.COMMON_ANCESTOR );

    // @public {Instance}
    this.filterRootInstance = filterRootInstance;

    // @private {Array.<SVGGradient>}
    this.dirtyGradients = cleanArray( this.dirtyGradients );

    // @private {Array.<SVGGroup>}
    this.dirtyGroups = cleanArray( this.dirtyGroups );

    // @private {Array.<Drawable>}
    this.dirtyDrawables = cleanArray( this.dirtyDrawables );

    // @private {CountMap.<Paint,SVGGradient|SVGPattern>}
    this.paintCountMap = this.paintCountMap || new CountMap(
      this.onAddPaint.bind( this ),
      this.onRemovePaint.bind( this )
    );

    // @private {boolean} - Tracks whether we have no dirty objects that would require cleanup or releases
    this.areReferencesReduced = true;

    if ( !this.domElement ) {

      // main SVG element
      this.svg = document.createElementNS( svgns, 'svg' );
      this.svg.style.position = 'absolute';
      this.svg.style.left = '0';
      this.svg.style.top = '0';

      // pdom - make sure the element is not focusable (it is focusable by default in IE11 full screen mode)
      this.svg.setAttribute( 'focusable', false );

      //OHTWO TODO: why would we clip the individual layers also? Seems like a potentially useless performance loss
      // this.svg.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
      this.svg.style[ 'pointer-events' ] = 'none';

      // @public {SVGDefsElement} - the <defs> block that we will be stuffing gradients and patterns into
      this.defs = document.createElementNS( svgns, 'defs' );
      this.svg.appendChild( this.defs );

      this.baseTransformGroup = document.createElementNS( svgns, 'g' );
      this.svg.appendChild( this.baseTransformGroup );

      this.domElement = this.svg;
    }

    // reset what layer fitting can do
    Utils.prepareForTransform( this.svg ); // Apply CSS needed for future CSS transforms to work properly.

    Utils.unsetTransform( this.svg ); // clear out any transforms that could have been previously applied
    this.baseTransformGroup.setAttribute( 'transform', '' ); // no base transform

    const instanceClosestToRoot = transformRootInstance.trail.nodes.length > filterRootInstance.trail.nodes.length ?
                                  filterRootInstance : transformRootInstance;

    this.rootGroup = SVGGroup.createFromPool( this, instanceClosestToRoot, null );
    this.baseTransformGroup.appendChild( this.rootGroup.svgGroup );

    // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)

    sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( `initialized #${this.id}` );

    return this;
  }

  /**
   * Callback for paintCountMap's create
   * @private
   *
   * @param {Paint} paint
   * @returns {SVGGradient|SVGPattern}
   */
  onAddPaint( paint ) {
    const svgPaint = paint.createSVGPaint( this );
    svgPaint.definition.setAttribute( 'id', `${paint.id}-${this.id}` );
    this.defs.appendChild( svgPaint.definition );

    return svgPaint;
  }

  /**
   * Callback for paintCountMap's destroy
   * @private
   *
   * @param {Paint} paint
   * @param {SVGGradient|SVGPattern} svgPaint
   */
  onRemovePaint( paint, svgPaint ) {
    this.defs.removeChild( svgPaint.definition );
    svgPaint.dispose();
  }

  /*
   * Increases our reference count for the specified {Paint}. If it didn't exist before, we'll add the SVG def to the
   * paint can be referenced by SVG id.
   * @public
   *
   * @param {Paint} paint
   */
  incrementPaint( paint ) {
    assert && assert( paint.isPaint );

    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `incrementPaint ${this} ${paint}` );

    this.paintCountMap.increment( paint );
  }

  /*
   * Decreases our reference count for the specified {Paint}. If this was the last reference, we'll remove the SVG def
   * from our SVG tree to prevent memory leaks, etc.
   * @public
   *
   * @param {Paint} paint
   */
  decrementPaint( paint ) {
    assert && assert( paint.isPaint );

    sceneryLog && sceneryLog.Paints && sceneryLog.Paints( `decrementPaint ${this} ${paint}` );

    this.paintCountMap.decrement( paint );
  }

  /**
   * @public
   *
   * @param {SVGGradient} gradient
   */
  markDirtyGradient( gradient ) {
    this.dirtyGradients.push( gradient );
    this.markDirty();
  }

  /**
   * @public
   *
   * @param {Block} block
   */
  markDirtyGroup( block ) {
    this.dirtyGroups.push( block );
    this.markDirty();

    if ( this.areReferencesReduced ) {
      this.display.markForReducedReferences( this );
    }
    this.areReferencesReduced = false;
  }

  /**
   * @public
   *
   * @param {Drawable} drawable
   */
  markDirtyDrawable( drawable ) {
    sceneryLog && sceneryLog.dirty && sceneryLog.dirty( `markDirtyDrawable on SVGBlock#${this.id} with ${drawable.toString()}` );
    this.dirtyDrawables.push( drawable );
    this.markDirty();

    if ( this.areReferencesReduced ) {
      this.display.markForReducedReferences( this );
    }
    this.areReferencesReduced = false;
  }

  /**
   * @public
   * @override
   */
  setSizeFullDisplay() {
    sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( `setSizeFullDisplay #${this.id}` );

    this.baseTransformGroup.removeAttribute( 'transform' );
    Utils.unsetTransform( this.svg );

    const size = this.display.getSize();
    this.svg.setAttribute( 'width', size.width );
    this.svg.setAttribute( 'height', size.height );
  }

  /**
   * @public
   * @override
   */
  setSizeFitBounds() {
    sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( `setSizeFitBounds #${this.id} with ${this.fitBounds.toString()}` );

    const x = this.fitBounds.minX;
    const y = this.fitBounds.minY;

    assert && assert( isFinite( x ) && isFinite( y ), 'Invalid SVG transform for SVGBlock' );
    assert && assert( this.fitBounds.isValid(), 'Invalid fitBounds' );

    this.baseTransformGroup.setAttribute( 'transform', `translate(${-x},${-y})` ); // subtract off so we have a tight fit
    Utils.setTransform( `matrix(1,0,0,1,${x},${y})`, this.svg ); // reapply the translation as a CSS transform
    this.svg.setAttribute( 'width', this.fitBounds.width );
    this.svg.setAttribute( 'height', this.fitBounds.height );
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

    sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( `update #${this.id}` );

    //OHTWO TODO: call here!
    // TODO: What does the above TODO mean?
    while ( this.dirtyGroups.length ) {
      const group = this.dirtyGroups.pop();

      // if this group has been disposed or moved to another block, don't mess with it
      if ( group.block === this ) {
        group.update();
      }
    }
    while ( this.dirtyGradients.length ) {
      this.dirtyGradients.pop().update();
    }
    while ( this.dirtyDrawables.length ) {
      const drawable = this.dirtyDrawables.pop();

      // if this drawable has been disposed or moved to another block, don't mess with it
      // TODO: If it was moved to another block, why might it still appear in our list?  Shouldn't that be an assertion check?
      if ( drawable.parentDrawable === this ) {
        drawable.update();
      }
    }

    this.areReferencesReduced = true; // Once we've iterated through things, we've automatically reduced our references.

    // checks will be done in updateFit() to see whether it is needed
    this.updateFit();

    return true;
  }

  /**
   * Looks to remove dirty objects that may have been disposed.
   * See https://github.com/phetsims/energy-forms-and-changes/issues/356
   * @public
   *
   * @public
   */
  reduceReferences() {
    // no-op if we had an update first
    if ( this.areReferencesReduced ) {
      return;
    }

    // Attempts to do this in a high-performance way, where we're not shifting array contents around (so we'll do this
    // in one scan).

    let inspectionIndex = 0;
    let replacementIndex = 0;

    while ( inspectionIndex < this.dirtyGroups.length ) {
      const group = this.dirtyGroups[ inspectionIndex ];

      // Only keep things that reference our block.
      if ( group.block === this ) {
        // If the indices are the same, don't do the operation
        if ( replacementIndex !== inspectionIndex ) {
          this.dirtyGroups[ replacementIndex ] = group;
        }
        replacementIndex++;
      }

      inspectionIndex++;
    }

    // Our array should be only that length now
    while ( this.dirtyGroups.length > replacementIndex ) {
      this.dirtyGroups.pop();
    }

    // Do a similar thing with dirtyDrawables (not optimized out because for right now we want to maximize performance).
    inspectionIndex = 0;
    replacementIndex = 0;

    while ( inspectionIndex < this.dirtyDrawables.length ) {
      const drawable = this.dirtyDrawables[ inspectionIndex ];

      // Only keep things that reference our block as the parentDrawable.
      if ( drawable.parentDrawable === this ) {
        // If the indices are the same, don't do the operation
        if ( replacementIndex !== inspectionIndex ) {
          this.dirtyDrawables[ replacementIndex ] = drawable;
        }
        replacementIndex++;
      }

      inspectionIndex++;
    }

    // Our array should be only that length now
    while ( this.dirtyDrawables.length > replacementIndex ) {
      this.dirtyDrawables.pop();
    }

    this.areReferencesReduced = true;
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( `dispose #${this.id}` );

    // make it take up zero area, so that we don't use up excess memory
    this.svg.setAttribute( 'width', '0' );
    this.svg.setAttribute( 'height', '0' );

    // clear references
    this.filterRootInstance = null;

    cleanArray( this.dirtyGradients );
    cleanArray( this.dirtyGroups );
    cleanArray( this.dirtyDrawables );

    this.paintCountMap.clear();

    this.baseTransformGroup.removeChild( this.rootGroup.svgGroup );
    this.rootGroup.dispose();
    this.rootGroup = null;

    // since we may not properly remove all defs yet
    while ( this.defs.childNodes.length ) {
      this.defs.removeChild( this.defs.childNodes[ 0 ] );
    }

    super.dispose();
  }

  /**
   * @public
   * @override
   *
   * @param {Drawable} drawable
   */
  addDrawable( drawable ) {
    sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( `#${this.id}.addDrawable ${drawable.toString()}` );

    super.addDrawable( drawable );

    SVGGroup.addDrawable( this, drawable );
    drawable.updateSVGBlock( this );
  }

  /**
   * @public
   * @override
   *
   * @param {Drawable} drawable
   */
  removeDrawable( drawable ) {
    sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( `#${this.id}.removeDrawable ${drawable.toString()}` );

    SVGGroup.removeDrawable( this, drawable );

    super.removeDrawable( drawable );

    // NOTE: we don't unset the drawable's defs here, since it will either be disposed (will clear it)
    // or will be added to another SVGBlock (which will overwrite it)
  }

  /**
   * @public
   * @override
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   */
  onIntervalChange( firstDrawable, lastDrawable ) {
    sceneryLog && sceneryLog.SVGBlock && sceneryLog.SVGBlock( `#${this.id}.onIntervalChange ${firstDrawable.toString()} to ${lastDrawable.toString()}` );

    super.onIntervalChange( firstDrawable, lastDrawable );
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    return `SVGBlock#${this.id}-${FittedBlock.fitString[ this.fit ]}`;
  }
}

scenery.register( 'SVGBlock', SVGBlock );

Poolable.mixInto( SVGBlock );

export default SVGBlock;