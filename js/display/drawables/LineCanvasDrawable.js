// Copyright 2016, University of Colorado Boulder

/**
 * Canvas drawable for Line nodes.
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
  // TODO: use LineStatelessDrawable instead of the custom stuff going on
  // var LineStatelessDrawable = require( 'SCENERY/display/drawables/LineStatelessDrawable' );

  /**
   * A generated CanvasSelfDrawable whose purpose will be drawing our Line. One of these drawables will be created
   * for each displayed instance of a Line.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function LineCanvasDrawable( renderer, instance ) {
    this.initializeCanvasSelfDrawable( renderer, instance );
    this.initializePaintableStateless( renderer, instance );
  }

  scenery.register( 'LineCanvasDrawable', LineCanvasDrawable );

  inherit( CanvasSelfDrawable, LineCanvasDrawable, {
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
      context.moveTo( node._x1, node._y1 );
      context.lineTo( node._x2, node._y2 );

      if ( node.hasPaintableStroke() ) {
        node.beforeCanvasStroke( wrapper ); // defined in Paintable
        context.stroke();
        node.afterCanvasStroke( wrapper ); // defined in Paintable
      }
    },

    // stateless dirty methods:
    markDirtyLine: function() { this.markPaintDirty(); },
    markDirtyP1: function() { this.markPaintDirty(); },
    markDirtyP2: function() { this.markPaintDirty(); },
    markDirtyX1: function() { this.markPaintDirty(); },
    markDirtyY1: function() { this.markPaintDirty(); },
    markDirtyX2: function() { this.markPaintDirty(); },
    markDirtyY2: function() { this.markPaintDirty(); },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      CanvasSelfDrawable.prototype.dispose.call( this );
      this.disposePaintableStateless();
    }
  } );

  PaintableStatelessDrawable.mixInto( LineCanvasDrawable );

  Poolable.mixInto( LineCanvasDrawable );

  return LineCanvasDrawable;
} );
