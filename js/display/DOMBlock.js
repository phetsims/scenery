// Copyright 2014-2022, University of Colorado Boulder

/**
 * DOM Drawable wrapper for another DOM Drawable. Used so that we can have our own independent siblings, generally as part
 * of a Backbone's layers/blocks.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../phet-core/js/Poolable.js';
import { Block, scenery } from '../imports.js';

class DOMBlock extends Block {
  /**
   * @mixes Poolable
   *
   * @param {Display} display
   * @param {Drawable} domDrawable
   */
  constructor( display, domDrawable ) {
    super();

    this.initialize( display, domDrawable );
  }

  /**
   * @public
   *
   * @param {Display} display
   * @param {Drawable} domDrawable
   * @returns {DOMBlock} - For chaining
   */
  initialize( display, domDrawable ) {
    // TODO: is it bad to pass the acceleration flags along?
    super.initialize( display, domDrawable.renderer );

    this.domDrawable = domDrawable;
    this.domElement = domDrawable.domElement;

    return this;
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

    this.domDrawable.update();

    return true;
  }

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    this.domDrawable = null;
    this.domElement = null;

    // super call
    super.dispose();
  }

  /**
   * @public
   *
   * @param {Drawable} drawable
   */
  markDirtyDrawable( drawable ) {
    this.markDirty();
  }

  /**
   * Adds a drawable to this block.
   * @public
   * @override
   *
   * @param {Drawable} drawable
   */
  addDrawable( drawable ) {
    sceneryLog && sceneryLog.DOMBlock && sceneryLog.DOMBlock( `#${this.id}.addDrawable ${drawable.toString()}` );
    assert && assert( this.domDrawable === drawable, 'DOMBlock should only be used with one drawable for now (the one it was initialized with)' );

    super.addDrawable( drawable );
  }

  /**
   * Removes a drawable from this block.
   * @public
   * @override
   *
   * @param {Drawable} drawable
   */
  removeDrawable( drawable ) {
    sceneryLog && sceneryLog.DOMBlock && sceneryLog.DOMBlock( `#${this.id}.removeDrawable ${drawable.toString()}` );
    assert && assert( this.domDrawable === drawable, 'DOMBlock should only be used with one drawable for now (the one it was initialized with)' );

    super.removeDrawable( drawable );
  }
}

scenery.register( 'DOMBlock', DOMBlock );

Poolable.mixInto( DOMBlock );

export default DOMBlock;