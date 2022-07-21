// Copyright 2016-2022, University of Colorado Boulder

/**
 * DOM drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { DOMSelfDrawable, ImageStatefulDrawable, scenery, Utils } from '../../imports.js';

// TODO: change this based on memory and performance characteristics of the platform
const keepDOMImageElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory

class ImageDOMDrawable extends ImageStatefulDrawable( DOMSelfDrawable ) {
  /**
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  constructor( renderer, instance ) {
    super( renderer, instance );

    // Apply CSS needed for future CSS transforms to work properly.
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
      this.domElement = document.createElement( 'img' );
      this.domElement.style.display = 'block';
      this.domElement.style.position = 'absolute';
      this.domElement.style.pointerEvents = 'none';
      this.domElement.style.left = '0';
      this.domElement.style.top = '0';
    }

    // Whether we have an opacity attribute specified on the DOM element.
    this.hasOpacity = false;
  }

  /**
   * Updates our DOM element so that its appearance matches our node's representation.
   * @protected
   *
   * This implements part of the DOMSelfDrawable required API for subtypes.
   */
  updateDOM() {
    const node = this.node;
    const img = this.domElement;

    if ( this.paintDirty && this.dirtyImage ) {
      // TODO: allow other ways of showing a DOM image?
      img.src = node._image ? node._image.src : '//:0'; // NOTE: for img with no src (but with a string), see http://stackoverflow.com/questions/5775469/whats-the-valid-way-to-include-an-image-with-no-src
    }

    if ( this.dirtyImageOpacity ) {
      if ( node._imageOpacity === 1 ) {
        if ( this.hasOpacity ) {
          this.hasOpacity = false;
          img.style.opacity = '';
        }
      }
      else {
        this.hasOpacity = true;
        img.style.opacity = node._imageOpacity;
      }
    }

    if ( this.transformDirty ) {
      Utils.applyPreparedTransform( this.getTransformMatrix(), this.domElement );
    }

    // clear all of the dirty flags
    this.setToCleanState();
    this.transformDirty = false;
  }

  /**
   * Disposes the drawable.
   * @public
   * @override
   */
  dispose() {
    if ( !keepDOMImageElements ) {
      this.domElement = null; // clear our DOM reference if we want to toss it
    }

    super.dispose();
  }
}

scenery.register( 'ImageDOMDrawable', ImageDOMDrawable );

Poolable.mixInto( ImageDOMDrawable );

export default ImageDOMDrawable;