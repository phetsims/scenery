// Copyright 2013-2021, University of Colorado Boulder

/**
 * DOM drawable for a single painted node.
 *
 * Subtypes should expose the following API that is used by DOMSelfDrawable:
 * - drawable.domElement {HTMLElement} - The primary DOM element that will get transformed and added.
 * - drawable.updateDOM() {function} - Called with no arguments in order to update the domElement's view.
 *
 * TODO: make abstract subtype methods for improved documentation
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, SelfDrawable } from '../imports.js';

class DOMSelfDrawable extends SelfDrawable {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   * @returns {DOMSelfDrawable}
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance );

    // @private {function} - this is the same across lifecycles
    this.transformListener = this.transformListener || this.markTransformDirty.bind( this );

    this.markTransformDirty();

    // @private {boolean}
    this.visibilityDirty = true;

    // handle transform changes
    instance.relativeTransform.addListener( this.transformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
    instance.relativeTransform.addPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated

    return this;
  }

  /**
   * @public
   */
  markTransformDirty() {
    // update the visual state available to updateDOM, so that it will update the transform (Text needs to change the transform, so it is included)
    this.transformDirty = true;

    this.markDirty();
  }

  /**
   * @public
   *
   * Called from the Node, probably during updateDOM
   *
   * @returns {Matrix3}
   */
  getTransformMatrix() {
    this.instance.relativeTransform.validate();
    return this.instance.relativeTransform.matrix;
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

    this.updateDOM();

    if ( this.visibilityDirty ) {
      this.visibilityDirty = false;

      this.domElement.style.visibility = this.visible ? '' : 'hidden';
    }

    this.cleanPaintableState && this.cleanPaintableState();

    return true;
  }

  /**
   * Called to update the visual appearance of our domElement
   * @protected
   * @abstract
   */
  updateDOM() {
    // should generally be overridden by drawable subtypes to implement the update
  }

  /**
   * @public
   * @override
   */
  updateSelfVisibility() {
    super.updateSelfVisibility();

    if ( !this.visibilityDirty ) {
      this.visibilityDirty = true;
      this.markDirty();
    }
  }

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    this.instance.relativeTransform.removeListener( this.transformListener );
    this.instance.relativeTransform.removePrecompute();

    // super call
    super.dispose();
  }
}

scenery.register( 'DOMSelfDrawable', DOMSelfDrawable );
export default DOMSelfDrawable;