// Copyright 2013-2021, University of Colorado Boulder

/**
 * Supertype for WebGL drawables that display a specific Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 */

import { scenery, SelfDrawable } from '../imports.js';

class WebGLSelfDrawable extends SelfDrawable {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   * @returns {WebGLSelfDrawable}
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance );

    // @private {function} - this is the same across lifecycles
    this.transformListener = this.transformListener || this.markTransformDirty.bind( this );

    // when our relative transform changes, notify us in the pre-repaint phase
    instance.relativeTransform.addListener( this.transformListener );

    // trigger precomputation of the relative transform, since we will always need it when it is updated
    instance.relativeTransform.addPrecompute();

    return this;
  }

  /**
   * @public
   */
  markTransformDirty() {
    this.markDirty();
  }

  /**
   * @public
   * @override
   */
  updateSelfVisibility() {
    super.updateSelfVisibility();

    // mark us as dirty when our self visibility changes
    this.markDirty();
  }

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    this.instance.relativeTransform.removeListener( this.transformListener );
    this.instance.relativeTransform.removePrecompute();

    super.dispose();
  }
}

scenery.register( 'WebGLSelfDrawable', WebGLSelfDrawable );

export default WebGLSelfDrawable;