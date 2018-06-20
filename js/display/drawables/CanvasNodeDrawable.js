// Copyright 2016, University of Colorado Boulder

/**
 * Canvas drawable for CanvasNode.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var ExperimentalPoolable = require( 'PHET_CORE/ExperimentalPoolable' );
  var inherit = require( 'PHET_CORE/inherit' );
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
    this.initialize( renderer, instance );
  }

  scenery.register( 'CanvasNodeDrawable', CanvasNodeDrawable );

  inherit( CanvasSelfDrawable, CanvasNodeDrawable, {
    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable trait (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @public (scenery-internal)
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     * @returns {CanvasNodeDrawable} - For chaining
     */
    initialize: function( renderer, instance ) {
      return this.initializeCanvasSelfDrawable( renderer, instance );
    },

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

  ExperimentalPoolable.mixInto( CanvasNodeDrawable, {
    initialize: CanvasNodeDrawable.prototype.initialize
  } );

  return CanvasNodeDrawable;
} );
