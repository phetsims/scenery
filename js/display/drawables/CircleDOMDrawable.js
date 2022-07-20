// Copyright 2016-2022, University of Colorado Boulder

/**
 * DOM drawable for Circle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { CircleStatefulDrawable, DOMSelfDrawable, Features, scenery, Utils } from '../../imports.js';

// TODO: change this based on memory and performance characteristics of the platform
const keepDOMCircleElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory

class CircleDOMDrawable extends CircleStatefulDrawable( DOMSelfDrawable ) {
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

    // @protected {Matrix3} - We need to store an independent matrix, as our CSS transform actually depends on the radius.
    this.matrix = this.matrix || Matrix3.pool.fetch();

    // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
    // allocation and performance costs)
    if ( !this.fillElement || !this.strokeElement ) {
      // @protected {HTMLDivElement} - Will contain the fill by manipulating borderRadius
      const fillElement = document.createElement( 'div' );
      this.fillElement = fillElement;

      // @protected {HTMLDivElement} - Will contain the stroke by manipulating borderRadius
      const strokeElement = document.createElement( 'div' );
      this.strokeElement = strokeElement;

      fillElement.style.display = 'block';
      fillElement.style.position = 'absolute';
      fillElement.style.left = '0';
      fillElement.style.top = '0';
      fillElement.style.pointerEvents = 'none';
      strokeElement.style.display = 'block';
      strokeElement.style.position = 'absolute';
      strokeElement.style.left = '0';
      strokeElement.style.top = '0';
      strokeElement.style.pointerEvents = 'none';

      // Nesting allows us to transform only one AND to guarantee that the stroke is on top.
      fillElement.appendChild( strokeElement );
    }

    // @protected {HTMLElement} - Our primary DOM element. This is exposed as part of the DOMSelfDrawable API.
    this.domElement = this.fillElement;
  }

  /**
   * Updates our DOM element so that its appearance matches our node's representation.
   * @protected
   *
   * This implements part of the DOMSelfDrawable required API for subtypes.
   */
  updateDOM() {
    const node = this.node;
    const fillElement = this.fillElement;
    const strokeElement = this.strokeElement;

    // If paintDirty is false, there are no updates that are needed.
    if ( this.paintDirty ) {
      if ( this.dirtyRadius ) {
        fillElement.style.width = `${2 * node._radius}px`;
        fillElement.style.height = `${2 * node._radius}px`;
        fillElement.style[ Features.borderRadius ] = `${node._radius}px`;
      }
      if ( this.dirtyFill ) {
        fillElement.style.backgroundColor = node.getCSSFill();
      }

      if ( this.dirtyStroke ) {
        // update stroke presence
        if ( node.hasStroke() ) {
          strokeElement.style.borderStyle = 'solid';
        }
        else {
          strokeElement.style.borderStyle = 'none';
        }
      }

      if ( node.hasStroke() ) {
        // since we only execute these if we have a stroke, we need to redo everything if there was no stroke previously.
        // the other option would be to update stroked information when there is no stroke (major performance loss for fill-only Circles)
        const hadNoStrokeBefore = !this.hadStroke;

        if ( hadNoStrokeBefore || this.dirtyLineWidth || this.dirtyRadius ) {
          strokeElement.style.width = `${2 * node._radius - node.getLineWidth()}px`;
          strokeElement.style.height = `${2 * node._radius - node.getLineWidth()}px`;
          strokeElement.style[ Features.borderRadius ] = `${node._radius + node.getLineWidth() / 2}px`;
        }
        if ( hadNoStrokeBefore || this.dirtyLineWidth ) {
          strokeElement.style.left = `${-node.getLineWidth() / 2}px`;
          strokeElement.style.top = `${-node.getLineWidth() / 2}px`;
          strokeElement.style.borderWidth = `${node.getLineWidth()}px`;
        }
        if ( hadNoStrokeBefore || this.dirtyStroke ) {
          strokeElement.style.borderColor = node.getSimpleCSSStroke();
        }
      }
    }

    // shift the element vertically, postmultiplied with the entire transform.
    if ( this.transformDirty || this.dirtyRadius ) {
      this.matrix.set( this.getTransformMatrix() );
      const translation = Matrix3.translation( -node._radius, -node._radius );
      this.matrix.multiplyMatrix( translation );
      translation.freeToPool();
      Utils.applyPreparedTransform( this.matrix, this.fillElement );
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
    // Release the DOM elements from the poolable visual state so they aren't kept in memory.
    // May not be done on platforms where we have enough memory to pool these
    if ( !keepDOMCircleElements ) {
      // clear the references
      this.fillElement = null;
      this.strokeElement = null;
      this.domElement = null;
    }

    super.dispose();
  }
}

scenery.register( 'CircleDOMDrawable', CircleDOMDrawable );

Poolable.mixInto( CircleDOMDrawable );

export default CircleDOMDrawable;