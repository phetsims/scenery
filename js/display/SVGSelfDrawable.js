// Copyright 2013-2025, University of Colorado Boulder

/**
 * Represents an SVG visual element, and is responsible for tracking changes to the visual element, and then applying
 * any changes at a later time.
 *
 * Abstract methods to implement for concrete implementations:
 *   updateSVGSelf() - Update the SVG element's state to what the Node's self should display
 *   updateDefsSelf( block ) - Update defs on the given block (or if block === null, remove)
 *   initializeState( renderer, instance )
 *   disposeState()
 *
 * Subtypes should also implement drawable.svgElement, as the actual SVG element to be used.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import arrayDifference from '../../../phet-core/js/arrayDifference.js';
import PaintSVGState from '../display/PaintSVGState.js';
import scenery from '../scenery.js';
import SelfDrawable from '../display/SelfDrawable.js';

// Scratch arrays so that we are not creating array objects during paint checks
const scratchOldPaints = [];
const scratchNewPaints = [];
const scratchSimilarPaints = [];
const emptyArray = [];

class SVGSelfDrawable extends SelfDrawable {
  /**
   * @public
   *
   * @param {number} renderer
   * @param {Instance} instance
   * @param {boolean} usesPaint - Effectively true if we mix in PaintableStatefulDrawable
   * @param {boolean} keepElements
   * @returns {SVGSelfDrawable}
   */
  initialize( renderer, instance, usesPaint, keepElements ) {
    assert && assert( typeof usesPaint === 'boolean' );
    assert && assert( typeof keepElements === 'boolean' );

    super.initialize( renderer, instance );

    // @private {boolean}
    this.usesPaint = usesPaint; // const! (always the same) - Effectively true if we mix in PaintableStatefulDrawable
    this.keepElements = keepElements; // const! (always the same)

    // @public {SVGElement} - should be filled in by subtype
    this.svgElement = null;

    // @public {SVGBlock} - will be updated by updateSVGBlock()
    this.svgBlock = null;

    if ( this.usesPaint ) {
      if ( !this.paintState ) {
        this.paintState = new PaintSVGState();
      }
      else {
        this.paintState.initialize();
      }
    }

    return this; // allow chaining
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

    this.updateSVG();

    return true;
  }

  /**
   * Updates the cached paints for this drawable. This should be called when the paints for the drawable have changed.
   * @protected
   */
  setCachedPaints( cachedPaints ) {
    assert && assert( this.usesPaint );
    assert && assert( Array.isArray( this.lastCachedPaints ) );

    // Fill scratch arrays with the differences between the last cached paints and the new cached paints
    arrayDifference( this.lastCachedPaints, cachedPaints, scratchOldPaints, scratchNewPaints, scratchSimilarPaints );

    // Increment paints first, so we DO NOT get rid of things we don't need.
    for ( let i = 0; i < scratchNewPaints.length; i++ ) {
      this.svgBlock.incrementPaint( scratchNewPaints[ i ] );
    }

    for ( let i = 0; i < scratchOldPaints.length; i++ ) {
      this.svgBlock.decrementPaint( scratchOldPaints[ i ] );
    }

    // Clear arrays so we don't temporarily leak memory (the next arrayDifference will push into these). Reduces
    // GC/created references.
    scratchOldPaints.length = 0;
    scratchNewPaints.length = 0;
    scratchSimilarPaints.length = 0;

    // Replace lastCachedPaints contents
    this.lastCachedPaints.length = 0;
    this.lastCachedPaints.push( ...cachedPaints );
  }

  /**
   * Called to update the visual appearance of our svgElement
   * @protected
   */
  updateSVG() {
    // sync the differences between the previously-recorded list of cached paints and the new list
    if ( this.usesPaint && this.dirtyCachedPaints ) {
      this.setCachedPaints( this.node._cachedPaints );
    }

    if ( this.paintDirty ) {
      this.updateSVGSelf( this.node, this.svgElement );
    }

    // clear all of the dirty flags
    this.setToCleanState();
  }

  /**
   * to be used by our passed in options.updateSVG
   * @protected
   *
   * @param {SVGElement} element
   */
  updateFillStrokeStyle( element ) {
    if ( !this.usesPaint ) {
      return;
    }

    if ( this.dirtyFill ) {
      this.paintState.updateFill( this.svgBlock, this.node.getFillValue() );
    }
    if ( this.dirtyStroke ) {
      this.paintState.updateStroke( this.svgBlock, this.node.getStrokeValue() );
    }
    const strokeDetailDirty = this.dirtyLineWidth || this.dirtyLineOptions;
    if ( strokeDetailDirty ) {
      this.paintState.updateStrokeDetailStyle( this.node );
    }
    if ( this.dirtyFill || this.dirtyStroke || strokeDetailDirty ) {
      element.setAttribute( 'style', this.paintState.baseStyle + this.paintState.strokeDetailStyle );
    }

    this.cleanPaintableState();
  }

  /**
   * @public
   *
   * @param {SVGBlock} svgBlock
   */
  updateSVGBlock( svgBlock ) {
    // remove cached paint references from the old svgBlock
    const oldSvgBlock = this.svgBlock;
    if ( this.usesPaint && oldSvgBlock ) {
      for ( let i = 0; i < this.lastCachedPaints.length; i++ ) {
        oldSvgBlock.decrementPaint( this.lastCachedPaints[ i ] );
      }
    }

    this.svgBlock = svgBlock;

    // add cached paint references from the new svgBlock
    if ( this.usesPaint ) {
      for ( let j = 0; j < this.lastCachedPaints.length; j++ ) {
        svgBlock.incrementPaint( this.lastCachedPaints[ j ] );
      }
    }

    this.updateDefsSelf && this.updateDefsSelf( svgBlock );

    this.usesPaint && this.paintState.updateSVGBlock( svgBlock );

    // since fill/stroke IDs may be block-specific, we need to mark them dirty so they will be updated
    this.usesPaint && this.markDirtyFill();
    this.usesPaint && this.markDirtyStroke();
  }

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    if ( !this.keepElements ) {
      // clear the references
      this.svgElement = null;
    }

    // release any defs, and dispose composed state objects
    this.updateDefsSelf && this.updateDefsSelf( null );

    if ( this.usesPaint ) {
      // When we are disposed, clear the cached paints (since we might switch to another node with a different set of
      // cached paints in the future). See https://github.com/phetsims/unit-rates/issues/226
      this.setCachedPaints( emptyArray );

      this.paintState.dispose();
    }

    this.defs = null;

    this.svgBlock = null;

    super.dispose();
  }
}

scenery.register( 'SVGSelfDrawable', SVGSelfDrawable );
export default SVGSelfDrawable;