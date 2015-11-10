// Copyright 2013-2015, University of Colorado Boulder

/**
 * A node that can be custom-drawn with Canvas calls. Manual handling of dirty region repainting.
 *
 * setCanvasBounds (or the mutator canvasBounds) should be used to set the area that is drawn to (otherwise nothing
 * will show up)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var Node = require( 'SCENERY/nodes/Node' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );

  var emptyArray = []; // constant

  // pass a canvasBounds option if you want to specify the self bounds
  function CanvasNode( options ) {
    Node.call( this, options );
    this.setRendererBitmask( Renderer.bitmaskCanvas );
  }
  scenery.register( 'CanvasNode', CanvasNode );

  inherit( Node, CanvasNode, {

    /**
     * How to set the bounds of the CanvasNode
     *
     * @param {Bounds2} selfBounds
     */
    setCanvasBounds: function( selfBounds ) {
      this.invalidateSelf( selfBounds );
    },
    set canvasBounds( value ) { this.setCanvasBounds( value ); },
    get canvasBounds() { return this.getSelfBounds(); },

    isPainted: function() {
      return true;
    },

    /**
     * Override paintCanvas with a faster version, since fillRect and drawRect don't affect the current default path.
     * @public
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

    invalidatePaint: function() {
      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirty();
      }
    },

    canvasPaintSelf: function( wrapper ) {
      this.paintCanvas( wrapper.context );
    },

    // override for computation of whether a point is inside the self content
    // point is considered to be in the local coordinate frame
    containsPointSelf: function( point ) {
      return false;
      // throw new Error( 'CanvasNode needs containsPointSelf implemented' );
    },

    createCanvasDrawable: function( renderer, instance ) {
      return CanvasNode.CanvasNodeDrawable.createFromPool( renderer, instance );
    },

    // whether this node's self intersects the specified bounds, in the local coordinate frame
    // intersectsBoundsSelf: function( bounds ) {
    //   // TODO: implement?
    // },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.CanvasNode( {' + propLines + '} )'; // TODO: no real way to do this nicely?
    }

  } );

  CanvasNode.prototype._mutatorKeys = [ 'canvasBounds' ].concat( Node.prototype._mutatorKeys );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  CanvasNode.CanvasNodeDrawable = function CanvasNodeDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( CanvasSelfDrawable, CanvasNode.CanvasNodeDrawable, {
    initialize: function( renderer, instance ) {
      return this.initializeCanvasSelfDrawable( renderer, instance );
    },

    paintCanvas: function( wrapper, node ) {
      assert && assert( !node.selfBounds.isEmpty(), 'CanvasNode should not be used with an empty canvasBounds. ' +
                                                    'Please set canvasBounds (or use setCanvasBounds()) on ' + node.constructor.name );

      if ( !node.selfBounds.isEmpty() ) {
        var context = wrapper.context;
        context.save();

        // set back to Canvas default styles
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
  SelfDrawable.Poolable.mixin( CanvasNode.CanvasNodeDrawable );

  return CanvasNode;
} );


