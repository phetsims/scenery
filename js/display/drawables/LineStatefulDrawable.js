// Copyright 2016-2022, University of Colorado Boulder

/**
 * A trait for drawables for Line that need to store state about what the current display is currently showing,
 * so that updates to the Line will only be made on attributes that specifically changed (and no change will be
 * necessary for an attribute that changed back to its original/currently-displayed value). Generally, this is used
 * for DOM and SVG drawables.
 *
 * This will also mix in PaintableStatefulDrawable
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import inheritance from '../../../../phet-core/js/inheritance.js';
import memoize from '../../../../phet-core/js/memoize.js';
import { PaintableStatefulDrawable, scenery, SelfDrawable } from '../../imports.js';

const LineStatefulDrawable = memoize( type => {
  assert && assert( _.includes( inheritance( type ), SelfDrawable ) );

  return class extends PaintableStatefulDrawable( type ) {
    /**
     * @public
     * @override
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     */
    initialize( renderer, instance, ...args ) {
      super.initialize( renderer, instance, ...args );

      // @protected {boolean} - Flag marked as true if ANY of the drawable dirty flags are set (basically everything except for transforms, as we
      //                        need to accelerate the transform case.
      this.paintDirty = true;
      this.dirtyX1 = true;
      this.dirtyY1 = true;
      this.dirtyX2 = true;
      this.dirtyY2 = true;
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
      this.dirtyX1 = true;
      this.dirtyY1 = true;
      this.dirtyX2 = true;
      this.dirtyY2 = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyP1() {
      this.dirtyX1 = true;
      this.dirtyY1 = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyP2() {
      this.dirtyX2 = true;
      this.dirtyY2 = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyX1() {
      this.dirtyX1 = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyY1() {
      this.dirtyY1 = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyX2() {
      this.dirtyX2 = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyY2() {
      this.dirtyY2 = true;
      this.markPaintDirty();
    }

    /**
     * Clears all of the dirty flags (after they have been checked), so that future mark* methods will be able to flag them again.
     * @public
     */
    setToCleanState() {
      this.paintDirty = false;
      this.dirtyX1 = false;
      this.dirtyY1 = false;
      this.dirtyX2 = false;
      this.dirtyY2 = false;
    }
  };
} );

scenery.register( 'LineStatefulDrawable', LineStatefulDrawable );
export default LineStatefulDrawable;