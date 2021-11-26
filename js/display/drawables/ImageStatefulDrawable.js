// Copyright 2016-2021, University of Colorado Boulder

/**
 * A trait for drawables for Image that need to store state about what the current display is currently showing,
 * so that updates to the Image will only be made on attributes that specifically changed (and no change will be
 * necessary for an attribute that changed back to its original/currently-displayed value). Generally, this is used
 * for DOM and SVG drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import inheritance from '../../../../phet-core/js/inheritance.js';
import memoize from '../../../../phet-core/js/memoize.js';
import { scenery, SelfDrawable } from '../../imports.js';

const ImageStatefulDrawable = memoize( type => {
  assert && assert( _.includes( inheritance( type ), SelfDrawable ) );

  return class extends type {
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
      this.dirtyImage = true;
      this.dirtyImageOpacity = true;
      this.dirtyMipmap = true;
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
    markDirtyImage() {
      this.dirtyImage = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyImageOpacity() {
      this.dirtyImageOpacity = true;
      this.markPaintDirty();
    }

    /**
     * @public
     */
    markDirtyMipmap() {
      this.dirtyMipmap = true;
      this.markPaintDirty();
    }

    /**
     * Clears all of the dirty flags (after they have been checked), so that future mark* methods will be able to flag them again.
     * @public
     */
    setToCleanState() {
      this.paintDirty = false;
      this.dirtyImage = false;
      this.dirtyImageOpacity = false;
      this.dirtyMipmap = false;
    }
  };
} );

scenery.register( 'ImageStatefulDrawable', ImageStatefulDrawable );

export default ImageStatefulDrawable;