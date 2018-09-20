// Copyright 2016, University of Colorado Boulder

/**
 * A trait for drawables for Circle that need to store state about what the current display is currently showing,
 * so that updates to the Circle will only be made on attributes that specifically changed (and no change will be
 * necessary for an attribute that changed back to its original/currently-displayed value). Generally, this is used
 * for DOM and SVG drawables.
 *
 * This trait assumes the PaintableStateful trait is also mixed (always the case for Circle stateful drawables).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inheritance = require( 'PHET_CORE/inheritance' );
  var PaintableStatefulDrawable = require( 'SCENERY/display/drawables/PaintableStatefulDrawable' );
  var scenery = require( 'SCENERY/scenery' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );

  var CircleStatefulDrawable = {
    /**
     * Given the type (constructor) of a drawable, we'll mix in a combination of:
     * - initialization/disposal with the *State suffix
     * - mark* methods to be called on all drawables of nodes of this type, that set specific dirty flags
     * @public (scenery-internal)
     *
     * This will allow drawables that mix in this type to do the following during an update:
     * 1. Check specific dirty flags (e.g. if the fill changed, update the fill of our SVG element).
     * 2. Call setToCleanState() once done, to clear the dirty flags.
     *
     * Also mixes in PaintableStatefulDrawable, as this is needed by all stateful Circle drawables.
     *
     * @param {function} drawableType - The constructor for the drawable type
     */
    mixInto: function( drawableType ) {
      assert && assert( _.includes( inheritance( drawableType ), SelfDrawable ) );

      var proto = drawableType.prototype;

      /**
       * Initializes the stateful trait state, starting its "lifetime" until it is disposed with disposeState().
       * @protected
       *
       * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
       * @param {Instance} instance
       * @returns {CircleStatefulDrawable} - Self reference for chaining
       */
      proto.initializeState = function( renderer, instance ) {
        // @protected {boolean} - Flag marked as true if ANY of the drawable dirty flags are set (basically everything except for transforms, as we
        //                        need to accelerate the transform case.
        this.paintDirty = true;

        // @protected {boolean} - Whether the radius has changed since our last update.
        this.dirtyRadius = true;

        // After adding flags, we'll initialize the mixed-in PaintableStateful state.
        this.initializePaintableState( renderer, instance );

        return this; // allow for chaining
      };

      /**
       * Disposes the stateful trait state, so it can be put into the pool to be initialized again.
       * @protected
       */
      proto.disposeState = function() {
        this.disposePaintableState();
      };

      /**
       * A "catch-all" dirty method that directly marks the paintDirty flag and triggers propagation of dirty
       * information. This can be used by other mark* methods, or directly itself if the paintDirty flag is checked.
       * @public (scenery-internal)
       *
       * It should be fired (indirectly or directly) for anything besides transforms that needs to make a drawable
       * dirty.
       */
      proto.markPaintDirty = function() {
        this.paintDirty = true;
        this.markDirty();
      };

      /**
       * Called when the radius of the circle changes.
       * @public (scenery-internal)
       */
      proto.markDirtyRadius = function() {
        this.dirtyRadius = true;
        this.markPaintDirty();
      };

      /**
       * Clears all of the dirty flags (after they have been checked), so that future mark* methods will be able to flag them again.
       * @public (scenery-internal)
       */
      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyRadius = false;
      };

      PaintableStatefulDrawable.mixInto( drawableType );
    }
  };

  scenery.register( 'CircleStatefulDrawable', CircleStatefulDrawable );

  return CircleStatefulDrawable;
} );
