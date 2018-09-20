// Copyright 2016, University of Colorado Boulder

/**
 * A trait for drawables for Line that need to store state about what the current display is currently showing,
 * so that updates to the Line will only be made on attributes that specifically changed (and no change will be
 * necessary for an attribute that changed back to its original/currently-displayed value). Generally, this is used
 * for DOM and SVG drawables.
 *
 * This trait assumes the PaintableStateful trait is also mixed (always the case for Line stateful drawables).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inheritance = require( 'PHET_CORE/inheritance' );
  var PaintableStatefulDrawable = require( 'SCENERY/display/drawables/PaintableStatefulDrawable' );
  var scenery = require( 'SCENERY/scenery' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );

  var LineStatefulDrawable = {
    /**
     * Given the type (constructor) of a drawable, we'll mix in a combination of:
     * - initialization/disposal with the *State suffix
     * - mark* methods to be called on all drawables of nodes of this type, that set specific dirty flags
     *
     * This will allow drawables that mix in this type to do the following during an update:
     * 1. Check specific dirty flags (e.g. if the fill changed, update the fill of our SVG element).
     * 2. Call setToCleanState() once done, to clear the dirty flags.
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
       * @returns {LineStatefulDrawable} - Self reference for chaining
       */
      proto.initializeState = function( renderer, instance ) {
        // @protected {boolean} - Flag marked as true if ANY of the drawable dirty flags are set (basically everything except for transforms, as we
        //                        need to accelerate the transform case.
        this.paintDirty = true;
        this.dirtyX1 = true;
        this.dirtyY1 = true;
        this.dirtyX2 = true;
        this.dirtyY2 = true;

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

      proto.markDirtyLine = function() {
        this.dirtyX1 = true;
        this.dirtyY1 = true;
        this.dirtyX2 = true;
        this.dirtyY2 = true;
        this.markPaintDirty();
      };

      proto.markDirtyP1 = function() {
        this.dirtyX1 = true;
        this.dirtyY1 = true;
        this.markPaintDirty();
      };

      proto.markDirtyP2 = function() {
        this.dirtyX2 = true;
        this.dirtyY2 = true;
        this.markPaintDirty();
      };

      proto.markDirtyX1 = function() {
        this.dirtyX1 = true;
        this.markPaintDirty();
      };

      proto.markDirtyY1 = function() {
        this.dirtyY1 = true;
        this.markPaintDirty();
      };

      proto.markDirtyX2 = function() {
        this.dirtyX2 = true;
        this.markPaintDirty();
      };

      proto.markDirtyY2 = function() {
        this.dirtyY2 = true;
        this.markPaintDirty();
      };

      /**
       * Clears all of the dirty flags (after they have been checked), so that future mark* methods will be able to flag them again.
       * @public (scenery-internal)
       */
      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyX1 = false;
        this.dirtyY1 = false;
        this.dirtyX2 = false;
        this.dirtyY2 = false;
      };

      PaintableStatefulDrawable.mixInto( drawableType );
    }
  };

  scenery.register( 'LineStatefulDrawable', LineStatefulDrawable );

  return LineStatefulDrawable;
} );
