// Copyright 2016-2022, University of Colorado Boulder

/**
 * A trait for drawables for Line that does not store the line's state, as it just needs to track dirtyness overall.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import inheritance from '../../../../phet-core/js/inheritance.js';
import memoize from '../../../../phet-core/js/memoize.js';
import { PaintableStatelessDrawable, scenery, SelfDrawable } from '../../imports.js';

const LineStatelessDrawable = memoize( type => {
  assert && assert( _.includes( inheritance( type ), SelfDrawable ) );

  return class extends PaintableStatelessDrawable( type ) {
    /**
     * @public
     * @override
     *
     * @param {number} renderer
     * @param {Instance} instance
     */
    initialize( renderer, instance, ...args ) {
      super.initialize( renderer, instance, ...args );

      // @protected {boolean} - Flag marked as true if ANY of the drawable dirty flags are set (basically everything except for transforms, as we
      //                        need to accelerate the transform case.
      this.paintDirty = true;
    }

    /**
     * A "catch-all" dirty method that directly marks the paintDirty flag and triggers propagation of dirty
     * information. This can be used by other mark* methods, or directly itself if the paintDirty flag is checked.
     * @public
     *
     * It should be fired (indirectly or directly) for anything besides transforms that needs to make a drawable
     * dirty.
     */
    markPaintDirty() {
      this.paintDirty = true;
      this.markDirty();
    }

    /**
     * @public
     */
    markDirtyLine() {
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyP1() {
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyP2() {
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyX1() {
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyY1() {
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyX2() {
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyY2() {
      this.markPaintDirty();
    }
  };
} );

scenery.register( 'LineStatelessDrawable', LineStatelessDrawable );
export default LineStatelessDrawable;