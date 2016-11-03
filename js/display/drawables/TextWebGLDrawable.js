// Copyright 2013-2015, University of Colorado Boulder

/**
 * WebGL drawable for Text nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Paintable = require( 'SCENERY/nodes/Paintable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );

  /**
   * A generated WebGLSelfDrawable whose purpose will be drawing our Text. One of these drawables will be created
   * for each displayed instance of a Text node.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function TextWebGLDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }

  scenery.register( 'TextWebGLDrawable', TextWebGLDrawable );

  inherit( WebGLSelfDrawable, TextWebGLDrawable, {
    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable mixin (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @public (scenery-internal)
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     * @returns {TextWebGLDrawable} - For chaining
     */
    initialize: function( renderer, instance ) {
      return this.initializeWebGLSelfDrawable( renderer, instance );
    },

    onAddToBlock: function( webglBlock ) {
      var self = this;
      this.node.toImageNodeAsynchronous( function( imageNodeContainer ) {
        //toImageNode returns a containerNode with its first child set as ImageNode
        var imageNode = imageNodeContainer.children[ 0 ];
        self.textHandle = webglBlock.webGLRenderer.textureRenderer.createFromImageNode( imageNode, 0.4 );

        // TODO: Don't call this each time a new item is added.
        webglBlock.webGLRenderer.textureRenderer.bindVertexBuffer();
        webglBlock.webGLRenderer.textureRenderer.bindDirtyTextures();
        self.updateText();
      } );

      //TODO: Update the state in the buffer arrays
    },

    onRemoveFromBlock: function( webglBlock ) {

    },

    //Nothing necessary since everything currently handled in the uModelViewMatrix below
    //However, we may switch to dynamic draw, and handle the matrix change only where necessary in the future?
    updateText: function() {
      if ( this.textHandle ) {
        this.textHandle.update();
      }
    },

    render: function( shaderProgram ) {
      // This is handled by the ColorTriangleRenderer
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      this.disposeWebGLBuffers();
      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    },

    disposeWebGLBuffers: function() {
      this.webglBlock.webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.dispose( this.rectangleHandle );
    },

    markDirtyText: function() {
      this.markDirty();
    },

    markDirtyBounds: function() {
      this.markDirty();
    },

    /**
     * A "catch-all" dirty method that directly marks the paintDirty flag and triggers propagation of dirty
     * information. This can be used by other mark* methods, or directly itself if the paintDirty flag is checked.
     * @public (scenery-internal)
     *
     * It should be fired (indirectly or directly) for anything besides transforms that needs to make a drawable
     * dirty.
     */
    markPaintDirty: function() {
      this.markDirty();
    },

    //TODO: Make sure all of the dirty flags make sense here.  Should we be using fillDirty, paintDirty, dirty, etc?
    update: function() {
      //if ( this.dirty ) {
      this.updateText();
      this.dirty = false;
      //}
    }
  } );

  // include stubs (stateless) for marking dirty stroke and fill (if necessary). we only want one dirty flag, not multiple ones, for WebGL (for now)
  Paintable.PaintableStatefulDrawable.mixin( TextWebGLDrawable );

  // This sets up TextWebGLDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( TextWebGLDrawable );

  return TextWebGLDrawable;
} );
