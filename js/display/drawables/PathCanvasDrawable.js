// Copyright 2016-2019, University of Colorado Boulder

/**
 * Canvas drawable for Path nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  const inherit = require( 'PHET_CORE/inherit' );
  const PaintableStatelessDrawable = require( 'SCENERY/display/drawables/PaintableStatelessDrawable' );
  const Poolable = require( 'PHET_CORE/Poolable' );
  const scenery = require( 'SCENERY/scenery' );

  /**
   * A generated CanvasSelfDrawable whose purpose will be drawing our Path. One of these drawables will be created
   * for each displayed instance of a Path.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function PathCanvasDrawable( renderer, instance ) {
    this.initializeCanvasSelfDrawable( renderer, instance );
    this.initializePaintableStateless( renderer, instance );
  }

  scenery.register( 'PathCanvasDrawable', PathCanvasDrawable );

  inherit( CanvasSelfDrawable, PathCanvasDrawable, {
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

      if ( node.hasShape() ) {
        // TODO: fill/stroke delay optimizations?
        context.beginPath();
        node._shape.writeToContext( context );

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
      }
    },

    // stateless dirty functions
    markDirtyShape: function() { this.markPaintDirty(); },

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

  PaintableStatelessDrawable.mixInto( PathCanvasDrawable );

  Poolable.mixInto( PathCanvasDrawable );

  return PathCanvasDrawable;
} );
