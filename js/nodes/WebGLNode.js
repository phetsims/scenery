// Copyright 2014-2015, University of Colorado Boulder

/**
 * A node that is drawn with custom WebGL calls, specified by the painter type passed in. Responsible for handling its
 * own bounds and invalidation (via setting canvasBounds and calling invalidatePaint()).
 *
 * This is the WebGL equivalent of CanvasNode.
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
  var Util = require( 'SCENERY/util/Util' );

  /**
   * @constructor
   *
   * It is required to pass a canvasBounds option and/or keep canvasBounds such that it will cover the entirety of the
   * Node. This will also set its self bounds.
   *
   * A "Painter" type should be passed to the constructor. It will be responsible for creating individual "painters"
   * that are used with different WebGL contexts to paint. This is helpful, since each context will need to have its
   * own buffers/textures/etc.
   *
   * painterType will be called with new painterType( gl, node ). Should contain the following methods:
   *
   * paint( modelViewMatrix, projectionMatrix )
   *   {Matrix3} modelViewMatrix - Transforms from the node's local coordinate frame to Scenery's global coordinate
   *                               frame.
   *   {Matrix3} projectionMatrix - Transforms from the global coordinate frame to normalized device coordinates.
   * dispose()
   *
   * @param {Function} painterType - The type (constructor) for the painters that will be used for this node.
   * @param {Object} [options]
   */
  function WebGLNode( painterType, options ) {
    Node.call( this, options );

    assert && assert( typeof painterType === 'function', 'Painter type now required by WebGLNode' );

    // Only support rendering in WebGL
    this.setRendererBitmask( Renderer.bitmaskWebGL );

    // @private {Function} - Used to create the painters
    this.painterType = painterType;
  }

  scenery.register( 'WebGLNode', WebGLNode );

  inherit( Node, WebGLNode, {
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
     * These bounds should always cover at least the area where the WebGLNode will draw in. If this is violated, this
     * node may be partially or completely invisible in Scenery's output.
     *
     * @param {Bounds2} selfBounds
     */
    setCanvasBounds: function( selfBounds ) {
      this.invalidateSelf( selfBounds );
    },
    set canvasBounds( value ) { this.setCanvasBounds( value ); },
    get canvasBounds() { return this.getSelfBounds(); },

    /**
     * Whether this Node itself is painted (displays something itself).
     * @public
     * @override
     *
     * @returns {boolean}
     */
    isPainted: function() {
      // Always true for WebGL nodes
      return true;
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
     * Computes whether the provided point is "inside" (contained) in this Node's self content, or "outside".
     * @protected
     * @override
     *
     * If WebGLNode subtypes want to support being picked or hit-tested, it should override this function.
     *
     * @param {Vector2} point - Considered to be in the local coordinate frame
     * @returns {boolean}
     */
    containsPointSelf: function( point ) {
      return false;
    },

    /**
     * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
     * coordinate frame of this node.
     * @protected
     * @override
     *
     * @param {CanvasContextWrapper} wrapper
     */
    canvasPaintSelf: function( wrapper ) {
      // TODO: see https://github.com/phetsims/scenery/issues/308
      assert && assert( 'unimplemented: canvasPaintSelf in WebGLNode' );
    },

    renderToCanvasSelf: function( wrapper, matrix ) {
      var width = wrapper.canvas.width;
      var height = wrapper.canvas.height;

      var scratchCanvas = document.createElement( 'canvas' );
      scratchCanvas.width = width;
      scratchCanvas.height = height;
      var contextOptions = {
        antialias: true,
        preserveDrawingBuffer: true // so we can get the data and render it to the Canvas
      };
      var gl = scratchCanvas.getContext( 'webgl', contextOptions ) || scratchCanvas.getContext( 'experimental-webgl', contextOptions );
      Util.applyWebGLContextDefaults( gl ); // blending, etc.

      var projectionMatrix = new Matrix3().setTo32Bit().rowMajor(
        2 / width, 0, -1,
        0, -2 / height, 1,
        0, 0, 1 );
      var modelViewMatrix = new Matrix3().setTo32Bit().set( matrix );
      gl.viewport( 0, 0, width, height );

      var PainterType = this.painterType;
      var painter = new PainterType( gl, this );

      painter.paint( modelViewMatrix, projectionMatrix );
      painter.dispose();

      gl.flush();

      wrapper.context.setTransform( 1, 0, 0, 1, 0, 0 ); // identity
      wrapper.context.drawImage( scratchCanvas, 0, 0 );
      wrapper.context.restore();
    },

    /**
     * Creates a WebGL drawable for this WebGLNode.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {WebGLSelfDrawable}
     */
    createWebGLDrawable: function( renderer, instance ) {
      return WebGLNode.WebGLNodeDrawable.createFromPool( renderer, instance );
    },

    /**
     * Returns a string containing constructor information for Node.string().
     * @protected
     * @override
     *
     * @param {string} propLines - A string representing the options properties that need to be set.
     * @returns {string}
     */
    getBasicConstructor: function( propLines ) {
      return 'new scenery.WebGLNode( {' + propLines + '} )'; // TODO: no real way to do this nicely?
    }
  }, {
    PAINTED_NOTHING: 0,
    PAINTED_SOMETHING: 1
  } );

  // Use a Float32Array-backed matrix, as it's better for usage with WebGL
  var modelViewMatrix = new Matrix3().setTo32Bit();

  WebGLNode.WebGLNodeDrawable = inherit( WebGLSelfDrawable, function WebGLNodeDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // What type of WebGL renderer/processor should be used.
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

      var painted = this.painter.paint( modelViewMatrix, this.webGLBlock.projectionMatrix );

      assert && assert( painted === WebGLNode.PAINTED_SOMETHING || painted === WebGLNode.PAINTED_NOTHING );
      assert && assert( WebGLNode.PAINTED_NOTHING === 0 && WebGLNode.PAINTED_SOMETHING === 1,
        'Ensure we can pass the value through directly to indicate whether draw calls were made' );

      return painted;
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
