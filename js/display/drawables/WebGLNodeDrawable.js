// Copyright 2016, University of Colorado Boulder

/**
 * WebGL drawable for WebGLNode.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );

  /**
   * A generated WebGLSelfDrawable whose purpose will be drawing our WebGLNode. One of these drawables will be created
   * for each displayed instance of a WebGLNode.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function WebGLNodeDrawable( renderer, instance ) {
    // @private {function}
    this.contextChangeListener = this.onWebGLContextChange.bind( this );

    // @private {*} - Will be set to whatever type node.painterType is.
    this.painter = null;

    this.initializeWebGLSelfDrawable( renderer, instance );
  }

  scenery.register( 'WebGLNodeDrawable', WebGLNodeDrawable );

  inherit( WebGLSelfDrawable, WebGLNodeDrawable, {
    // What type of WebGL renderer/processor should be used. TODO: doc
    webglRenderer: Renderer.webglCustom,

    /**
     * Creates an instance of our Node's "painter" type.
     * @private
     *
     * @returns {*} - Whatever node.painterType is will be the type.
     */
    createPainter: function() {
      var PainterType = this.node.painterType;
      return new PainterType( this.webGLBlock.gl, this.node );
    },

    /**
     * Callback for when the WebGL context changes. We'll reconstruct the painter.
     * @public (scenery-internal)
     */
    onWebGLContextChange: function() {
      //TODO: Should a function be added for "disposeNonWebGL"?

      // Create the new painter
      this.painter = this.createPainter();
    },

    onAddToBlock: function( webGLBlock ) {
      // @private {WebGLBlock}
      this.webGLBlock = webGLBlock;

      this.painter = this.createPainter();

      webGLBlock.glChangedEmitter.addListener( this.contextChangeListener );
    },

    onRemoveFromBlock: function( webGLBlock ) {
      webGLBlock.glChangedEmitter.removeListener( this.contextChangeListener );
    },

    draw: function() {
      // we have a precompute need
      var matrix = this.instance.relativeTransform.matrix;

      var painted = this.painter.paint( matrix, this.webGLBlock.projectionMatrix );

      assert && assert( painted === scenery.WebGLNode.PAINTED_SOMETHING || painted === scenery.WebGLNode.PAINTED_NOTHING );
      assert && assert( scenery.WebGLNode.PAINTED_NOTHING === 0 && scenery.WebGLNode.PAINTED_SOMETHING === 1,
        'Ensure we can pass the value through directly to indicate whether draw calls were made' );

      return painted;
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      this.painter.dispose();
      this.painter = null;

      if ( this.webGLBlock ) {
        this.webGLBlock = null;
      }

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
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

    // forward call to the WebGLNode
    get shaderAttributes() {
      return this.node.shaderAttributes;
    }
  } );

  Poolable.mixInto( WebGLNodeDrawable );

  return WebGLNodeDrawable;
} );
