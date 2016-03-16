// Copyright 2014-2015, University of Colorado Boulder

/**
 * A node that can be custom-drawn with WebGL calls. Manual handling of dirty region repainting.  Analogous to CanvasNode
 *
 * setCanvasBounds (or the mutator canvasBounds) should be used to set the area that is drawn to (otherwise nothing
 * will show up)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );

  // pass a canvasBounds option if you want to specify the self bounds
  function WebGLNode( painterType, options ) {
    Node.call( this, options );
    this.setRendererBitmask( Renderer.bitmaskWebGL );

    /**
     * @private {Function}
     *
     * painterType will be called with new painterType( gl, node ). Should contain the following methods:
     *
     * paint( modelViewMatrix, projectionMatrix )
     *   {Matrix3} modelViewMatrix - Transforms from the node's local coordinate frame to Scenery's global coordinate
     *                               frame.
     *   {Matrix3} projectionMatrix - Transforms from the global coordinate frame to normalized device coordinates.
     *
     * dispose()
     */
    this.painterType = painterType;
  }

  scenery.register( 'WebGLNode', WebGLNode );

  inherit( Node, WebGLNode, {

    // how to set the bounds of the WebGLNode
    setCanvasBounds: function( selfBounds ) {
      this.invalidateSelf( selfBounds );
    },
    set canvasBounds( value ) { this.setCanvasBounds( value ); },
    get canvasBounds() { return this.getSelfBounds(); },

    isPainted: function() {
      return true;
    },

    /**
     * Initializes a WebGL drawable for a displayed instance of this node.
     * @public
     *
     * Meant to be overridden by a concrete sub-type.
     *
     * IMPORTANT NOTE: This function will be run from inside Scenery's Display.updateDisplay(), so it should not modify
     * or mutate any Scenery nodes (particularly anything that would cause something to be marked as needing a repaint).
     * Ideally, this function should have no outside effects other than painting to the Canvas provided.
     *
     * @param {WebGLNode.WebGLNodeDrawable} drawable
     */
    initializeWebGLDrawable: function( drawable ) {
      throw new Error( 'WebGLNode needs initializeWebGLDrawable implemented' );
    },

    /**
     * Paints a WebGL drawable for a displayed instance of this node.
     * @public
     *
     * Meant to be overridden by a concrete sub-type.
     *
     * IMPORTANT NOTE: This function will be run from inside Scenery's Display.updateDisplay(), so it should not modify
     * or mutate any Scenery nodes (particularly anything that would cause something to be marked as needing a repaint).
     * Ideally, this function should have no outside effects other than painting to the Canvas provided.
     *
     * For handling transforms, this function provides a matrix with the local-to-global coordinate transform, e.g.:
     * gl.uniformMatrix3fv( uniforms.uModelViewMatrix, false, matrix.entries );
     * AND also a recommended projection transform:
     * gl.uniformMatrix3fv( uniforms.uProjectionMatrix, false, drawable.webGLBlock.projectionMatrixArray );
     *
     * @param {WebGLNode.WebGLNodeDrawable} drawable
     * @param {Matrix3} matrix - The model-view matrix, from this node's local coordinate frame to Scenery's
     *                           global coordinate frame
     */
    paintWebGLDrawable: function( drawable, matrix ) {
      throw new Error( 'WebGLNode needs paintWebGLDrawable implemented' );
    },

    /**
     * Cleans up a WebGL drawable for a displayed instance of this node.
     * @public
     *
     * Meant to be overridden by a concrete sub-type.
     *
     * IMPORTANT NOTE: This function will be run from inside Scenery's Display.updateDisplay(), so it should not modify
     * or mutate any Scenery nodes (particularly anything that would cause something to be marked as needing a repaint).
     * Ideally, this function should have no outside effects other than painting to the Canvas provided.
     *
     * @param {WebGLNode.WebGLNodeDrawable} drawable
     */
    disposeWebGLDrawable: function( drawable ) {
      throw new Error( 'WebGLNode needs disposeWebGLDrawable implemented' );
    },

    invalidatePaint: function() {
      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirty();
      }
    },

    // override for computation of whether a point is inside the self content
    // point is considered to be in the local coordinate frame
    containsPointSelf: function( point ) {
      return false;
      // throw new Error( 'WebGLNode needs containsPointSelf implemented' );
    },

    canvasPaintSelf: function( wrapper ) {
      assert && assert( 'unimplemented: canvasPaintSelf in WebGLNode' );
    },

    createWebGLDrawable: function( renderer, instance ) {
      return WebGLNode.WebGLNodeDrawable.createFromPool( renderer, instance );
    },

    // whether this node's self intersects the specified bounds, in the local coordinate frame
    // intersectsBoundsSelf: function( bounds ) {
    //   // TODO: implement?
    // },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.WebGLNode( {' + propLines + '} )'; // TODO: no real way to do this nicely?
    }

  } );

  WebGLNode.prototype._mutatorKeys = [ 'canvasBounds' ].concat( Node.prototype._mutatorKeys );

  var modelViewMatrix = new Matrix3().setTo32Bit();

  WebGLNode.WebGLNodeDrawable = inherit( WebGLSelfDrawable, function WebGLNodeDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    webglRenderer: Renderer.webglCustom,

    // called either from the constructor, or from pooling
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );
    },

    onAddToBlock: function( webGLBlock ) {
      this.webGLBlock = webGLBlock;
      this.backingScale = this.webGLBlock.backingScale;
      this.gl = this.webGLBlock.gl;

      var PainterType = this.node.painterType;
      this.painter = new PainterType( this.gl, this.node );
    },

    onRemoveFromBlock: function( webGLBlock ) {

    },

    draw: function() {
      // we have a precompute need
      var matrix = this.instance.relativeTransform.matrix;

      modelViewMatrix.set( matrix );

      this.painter.paint( modelViewMatrix, this.webGLBlock.projectionMatrix );
    },

    dispose: function() {
      this.painter.dispose();

      if ( this.webGLBlock ) {
        this.webGLBlock = null;
      }

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    },

    // general flag set on the state, which we forward directly to the drawable's paint flag
    markPaintDirty: function() {
      this.markDirty();
    },

    // forward call to the WebGLNode
    get shaderAttributes() {
      return this.node.shaderAttributes;
    },

    update: function() {
      this.dirty = false;
    }
  } );
  SelfDrawable.Poolable.mixin( WebGLNode.WebGLNodeDrawable ); // pooling

  return WebGLNode;
} );
