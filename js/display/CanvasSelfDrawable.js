// Copyright 2013-2021, University of Colorado Boulder

/**
 * TODO docs
 *   note paintCanvas() required, and other implementation-specific details
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, SelfDrawable } from '../imports.js';

class CanvasSelfDrawable extends SelfDrawable {
  /**
   * @public
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance );

    // @private {function} - this is the same across lifecycles
    this.transformListener = this.transformListener || this.markTransformDirty.bind( this );

    instance.relativeTransform.addListener( this.transformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
    instance.relativeTransform.addPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated
  }

  /**
   * @public
   */
  markTransformDirty() {
    this.markDirty();
  }

  /**
   * General flag set on the state, which we forward directly to the drawable's paint flag
   * @public
   */
  markPaintDirty() {
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

scenery.register( 'CanvasSelfDrawable', CanvasSelfDrawable );

export default CanvasSelfDrawable;