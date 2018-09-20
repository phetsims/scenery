// Copyright 2016, University of Colorado Boulder

/**
 * Canvas drawable for Circle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var PaintableStatelessDrawable = require( 'SCENERY/display/drawables/PaintableStatelessDrawable' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * A generated CanvasSelfDrawable whose purpose will be drawing our Circle. One of these drawables will be created
   * for each displayed instance of a Circle.
   * @public (scenery-internal)
   * @constructor
   * @extends CanvasSelfDrawable
   * @mixes PaintableStatelessDrawable
   * @mixes SelfDrawable.Poolable
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function CircleCanvasDrawable( renderer, instance ) {
    this.initializeCanvasSelfDrawable( renderer, instance );
    this.initializePaintableStateless( renderer, instance );
  }

  scenery.register( 'CircleCanvasDrawable', CircleCanvasDrawable );

  inherit( CanvasSelfDrawable, CircleCanvasDrawable, {
    /**
     * Paints this drawable to a Canvas (the wrapper contains both a Canvas reference and its drawing context).
     * @public
     *
     * Assumes that the Canvas's context is already in the proper local coordinate frame for the node, and that any
     * other required effects (opacity, clipping, etc.) have already been prepared.
     *
     * This is part of the CanvasSelfDrawable API required to be implemented for subtypes.
     *
     * @param {CanvasContextWrapper} wrapper - Contains the Canvas and its drawing context
     * @param {Node} node - Our node that is being drawn
     * @param {Matrix3} matrix - The transformation matrix applied for this node's coordinate system.
     */
    paintCanvas: function( wrapper, node, matrix ) {
      var context = wrapper.context;

      context.beginPath();
      context.arc( 0, 0, node._radius, 0, Math.PI * 2, false );
      context.closePath();

      if ( node.hasFill() ) {
        node.beforeCanvasFill( wrapper ); // defined in Paintable
        context.fill();
        node.afterCanvasFill( wrapper ); // defined in Paintable
      }
      if ( node.hasPaintableStroke() ) {
        node.beforeCanvasStroke( wrapper ); // defined in Paintable
        context.stroke();
        node.afterCanvasStroke( wrapper ); // defined in Paintable
      }
    },

    /**
     * Called when the radius of the circle changes.
     * @public (scenery-internal)
     */
    markDirtyRadius: function() {
      this.markPaintDirty();
    },

    /**
     * Disposes the drawable.
     * @public (scenery-internal)
     * @override
     */
    dispose: function() {
      CanvasSelfDrawable.prototype.dispose.call( this );
      this.disposePaintableStateless();
    }
  } );

  // Since we're not using Circle's stateful trait, we'll need to mix in the Paintable trait here (of the stateless variety).
  PaintableStatelessDrawable.mixInto( CircleCanvasDrawable );

  Poolable.mixInto( CircleCanvasDrawable );

  return CircleCanvasDrawable;
} );
