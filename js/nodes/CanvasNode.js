// Copyright 2013-2016, University of Colorado Boulder

/**
 * An abstract node (should be subtyped) that is drawn by user-provided custom Canvas code.
 *
 * The region that can be drawn in is handled manually, by controlling the canvasBounds property of this CanvasNode.
 * Any regions outside of the canvasBounds will not be guaranteed to be drawn. This can be set with canvasBounds in the
 * constructor, or later with node.canvasBounds = bounds or setCanvasBounds( bounds ).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var CanvasNodeDrawable = require( 'SCENERY/display/drawables/CanvasNodeDrawable' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Renderer = require( 'SCENERY/display/Renderer' );

  /**
   * @public
   * @constructor
   * @extends Node
   *
   * @param {Object} [options] - Can contain Node's options, and/or CanvasNode options (e.g. canvasBounds)
   */
  function CanvasNode( options ) {
    Node.call( this, options );

    // This shouldn't change, as we only support one renderer
    this.setRendererBitmask( Renderer.bitmaskCanvas );
  }

  scenery.register( 'CanvasNode', CanvasNode );

  inherit( Node, CanvasNode, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: [ 'canvasBounds' ].concat( Node.prototype._mutatorKeys ),

    /**
     * Sets the bounds that are used for layout/repainting.
     * @public
     *
     * These bounds should always cover at least the area where the CanvasNode will draw in. If this is violated, this
     * node may be partially or completely invisible in Scenery's output.
     *
     * @param {Bounds2} selfBounds
     */
    setCanvasBounds: function( selfBounds ) {
      this.invalidateSelf( selfBounds );
    },
    set canvasBounds( value ) { this.setCanvasBounds( value ); },

    /**
     * Returns the previously-set canvasBounds, or Bounds2.NOTHING if it has not been set yet.
     * @public
     *
     * @returns {Bounds2}
     */
    getCanvasBounds: function() {
      return this.getSelfBounds();
    },
    get canvasBounds() { return this.getCanvasBounds(); },

    /**
     * Whether this Node itself is painted (displays something itself).
     * @public
     * @override
     *
     * @returns {boolean}
     */
    isPainted: function() {
      // Always true for CanvasNode
      return true;
    },

    /**
     * Override paintCanvas with a faster version, since fillRect and drawRect don't affect the current default path.
     * @public
     * @abstract
     *
     * IMPORTANT NOTE: This function will be run from inside Scenery's Display.updateDisplay(), so it should not modify
     * or mutate any Scenery nodes (particularly anything that would cause something to be marked as needing a repaint).
     * Ideally, this function should have no outside effects other than painting to the Canvas provided.
     *
     * @param {CanvasRenderingContext2D} context
     */
    paintCanvas: function( context ) {
      throw new Error( 'CanvasNode needs paintCanvas implemented' );
    },

    /**
     * Should be called when this node needs to be repainted. When not called, Scenery assumes that this node does
     * NOT need to be repainted (although Scenery may repaint it due to other nodes needing to be repainted).
     * @public
     *
     * This sets a "dirty" flag, so that it will be repainted the next time it would be displayed.
     */
    invalidatePaint: function() {
      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirty();
      }
    },

    /**
     * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
     * coordinate frame of this node.
     * @protected
     * @override
     *
     * @param {CanvasContextWrapper} wrapper
     * @param {Matrix3} matrix - The transformation matrix already applied to the context.
     */
    canvasPaintSelf: function( wrapper, matrix ) {
      this.paintCanvas( wrapper.context );
    },

    /**
     * Computes whether the provided point is "inside" (contained) in this Node's self content, or "outside".
     * @protected
     * @override
     *
     * If CanvasNode subtypes want to support being picked or hit-tested, it should override this function.
     *
     * @param {Vector2} point - Considered to be in the local coordinate frame
     * @returns {boolean}
     */
    containsPointSelf: function( point ) {
      return false;
    },

    /**
     * Creates a Canvas drawable for this CanvasNode.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {CanvasSelfDrawable}
     */
    createCanvasDrawable: function( renderer, instance ) {
      return CanvasNodeDrawable.createFromPool( renderer, instance );
    }
  } );

  return CanvasNode;
} );
