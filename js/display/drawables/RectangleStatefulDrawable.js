// Copyright 2016-2022, University of Colorado Boulder

/**
 * A trait for drawables for Rectangle that need to store state about what the current display is currently showing,
 * so that updates to the Rectangle will only be made on attributes that specifically changed (and no change will be
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

const RectangleStatefulDrawable = memoize( type => {
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
      this.dirtyX = true;
      this.dirtyY = true;
      this.dirtyWidth = true;
      this.dirtyHeight = true;
      this.dirtyCornerXRadius = true;
      this.dirtyCornerYRadius = true;
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
    markDirtyRectangle() {
      // TODO: consider bitmask instead?
      this.dirtyX = true;
      this.dirtyY = true;
      this.dirtyWidth = true;
      this.dirtyHeight = true;
      this.dirtyCornerXRadius = true;
      this.dirtyCornerYRadius = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyX() {
      this.dirtyX = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyY() {
      this.dirtyY = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyWidth() {
      this.dirtyWidth = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyHeight() {
      this.dirtyHeight = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyCornerXRadius() {
      this.dirtyCornerXRadius = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyCornerYRadius() {
      this.dirtyCornerYRadius = true;
      this.markPaintDirty();
    }

    /**
     * Clears all of the dirty flags (after they have been checked), so that future mark* methods will be able to flag them again.
     * @public
     */
    setToCleanState() {
      this.paintDirty = false;
      this.dirtyX = false;
      this.dirtyY = false;
      this.dirtyWidth = false;
      this.dirtyHeight = false;
      this.dirtyCornerXRadius = false;
      this.dirtyCornerYRadius = false;
    }
  };
} );

scenery.register( 'RectangleStatefulDrawable', RectangleStatefulDrawable );

export default RectangleStatefulDrawable;