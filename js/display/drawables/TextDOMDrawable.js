// Copyright 2016-2022, University of Colorado Boulder

/**
 * DOM drawable for Text nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { DOMSelfDrawable, scenery, TextStatefulDrawable, Utils } from '../../imports.js';

// TODO: change this based on memory and performance characteristics of the platform
const keepDOMTextElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory

// scratch matrix used in DOM rendering
const scratchMatrix = Matrix3.pool.fetch();

class TextDOMDrawable extends TextStatefulDrawable( DOMSelfDrawable ) {
  /**
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  constructor( renderer, instance ) {
    super( renderer, instance );

    // Apply CSS needed for future CSS transforms to work properly. Just do this once for performance
    Utils.prepareForTransform( this.domElement );
  }

  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance );

    // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
    // allocation and performance costs)
    if ( !this.domElement ) {
      // @protected {HTMLElement} - Our primary DOM element. This is exposed as part of the DOMSelfDrawable API.
      this.domElement = document.createElement( 'div' );
      this.domElement.style.display = 'block';
      this.domElement.style.position = 'absolute';
      this.domElement.style.pointerEvents = 'none';
      this.domElement.style.left = '0';
      this.domElement.style.top = '0';
      this.domElement.setAttribute( 'dir', 'ltr' );
    }
  }

  /**
   * Updates our DOM element so that its appearance matches our node's representation.
   * @protected
   *
   * This implements part of the DOMSelfDrawable required API for subtypes.
   */
  updateDOM() {
    const node = this.node;

    const div = this.domElement;

    if ( this.paintDirty ) {
      if ( this.dirtyFont ) {
        div.style.font = node.getFont();
      }
      if ( this.dirtyStroke ) {
        div.style.color = node.getCSSFill();
      }
      if ( this.dirtyBounds ) { // TODO: this condition is set on invalidateText, so it's almost always true?
        div.style.width = `${node.getSelfBounds().width}px`;
        div.style.height = `${node.getSelfBounds().height}px`;
        // TODO: do we require the jQuery versions here, or are they vestigial?
        // $div.width( node.getSelfBounds().width );
        // $div.height( node.getSelfBounds().height );
      }
      if ( this.dirtyText ) {
        div.textContent = node.renderedText;
      }
    }

    if ( this.transformDirty || this.dirtyText || this.dirtyFont || this.dirtyBounds ) {
      // shift the text vertically, postmultiplied with the entire transform.
      const yOffset = node.getSelfBounds().minY;
      scratchMatrix.set( this.getTransformMatrix() );
      const translation = Matrix3.translation( 0, yOffset );
      scratchMatrix.multiplyMatrix( translation );
      translation.freeToPool();
      Utils.applyPreparedTransform( scratchMatrix, div );
    }

    // clear all of the dirty flags
    this.setToCleanState();
    this.cleanPaintableState();
    this.transformDirty = false;
  }

  /**
   * Disposes the drawable.
   * @public
   * @override
   */
  dispose() {
    if ( !keepDOMTextElements ) {
      // clear the references
      this.domElement = null;
    }

    super.dispose();
  }
}

scenery.register( 'TextDOMDrawable', TextDOMDrawable );

Poolable.mixInto( TextDOMDrawable );

export default TextDOMDrawable;