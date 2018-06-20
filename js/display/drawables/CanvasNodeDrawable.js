// Copyright 2016, University of Colorado Boulder

/**
 * Canvas drawable for CanvasNode.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );

  var emptyArray = []; // constant, used for line-dash

  /**
   * A generated CanvasSelfDrawable whose purpose will be drawing our CanvasNode. One of these drawables will be created
   * for each displayed instance of a CanvasNode.
   * @public (scenery-internal)
   * @constructor
   * @extends CanvasSelfDrawable
   * @mixes SelfDrawable.Poolable
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function CanvasNodeDrawable( renderer, instance ) {
    this.initializeCanvasSelfDrawable( renderer, instance );
  }

  scenery.register( 'CanvasNodeDrawable', CanvasNodeDrawable );

  inherit( CanvasSelfDrawable, CanvasNodeDrawable, {
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
      assert && assert( !node.selfBounds.isEmpty(), 'CanvasNode should not be used with an empty canvasBounds. ' +
                                                    'Please set canvasBounds (or use setCanvasBounds()) on ' + node.constructor.name );

      if ( !node.selfBounds.isEmpty() ) {
        var context = wrapper.context;
        context.save();

        // set back to Canvas default styles
        // TODO: are these necessary, or can we drop them for performance?
        context.fillStyle = 'black';
        context.strokeStyle = 'black';
        context.lineWidth = 1;
        context.lineCap = 'butt';
        context.lineJoin = 'miter';
        context.lineDash = emptyArray;
        context.lineDashOffset = 0;
        context.miterLimit = 10;

        node.paintCanvas( context );

        context.restore();
      }
    }
  } );

  Poolable.mixInto( CanvasNodeDrawable );

  return CanvasNodeDrawable;
} );
