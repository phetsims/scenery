// Copyright 2014-2016, University of Colorado Boulder

/**
 * An abstract node (should be subtyped) that is drawn by user-provided custom WebGL code.
 *
 * The region that can be drawn in is handled manually, by controlling the canvasBounds property of this WebGLNode.
 * Any regions outside of the canvasBounds will not be guaranteed to be drawn. This can be set with canvasBounds in the
 * constructor, or later with node.canvasBounds = bounds or setCanvasBounds( bounds ).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var Util = require( 'SCENERY/util/Util' );
  var WebGLNodeDrawable = require( 'SCENERY/display/drawables/WebGLNodeDrawable' );

  var WEBGL_NODE_OPTION_KEYS = [
    'canvasBounds' // Sets the available Canvas bounds that content will show up in. See setCanvasBounds()
  ];

  /**
   * @public
   * @constructor
   * @extends Node
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
   *   Returns either WebGLNode.PAINTED_NOTHING or WebGLNode.PAINTED_SOMETHING.
   * dispose()
   *
   * NOTE: If any alpha values are non-1, please note that Scenery's canvases uses blending/settings for premultiplied
   * alpha. This means that if you want a color to look like (r,g,b,a), the value passed to gl_FragColor should be
   * (r/a,g/a,b/a,a).
   *
   * @param {Function} painterType - The type (constructor) for the painters that will be used for this node.
   * @param {Object} [options] - WebGLNode-specific options are documented in LINE_OPTION_KEYS above, and can be
   *                             provided along-side options for Node
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
    _mutatorKeys: WEBGL_NODE_OPTION_KEYS.concat( Node.prototype._mutatorKeys ),

    /**
     * Sets the bounds that are used for layout/repainting.
     * @public
     *
     * These bounds should always cover at least the area where the WebGLNode will draw in. If this is violated, this
     * node may be partially or completely invisible in Scenery's output.
     *
     * @param {Bounds2} selfBounds
     * @returns {WebGLNode} - For Chaining
     */
    setCanvasBounds: function( selfBounds ) {
      this.invalidateSelf( selfBounds );

      return this;
    },
    set canvasBounds( value ) { this.setCanvasBounds( value ); },

    /**
     * Returns the previously-set canvasBounds, or Bounds2.NOTHING if it has not been set yet.
     * @public
     *
     * @returns {Bounds2}
     */
    getCanvasBounds: function() {
      return this.getSelfBounds();
    },
    get canvasBounds() { return this.getCanvasBounds(); },


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
     * @param {Matrix3} matrix - The transformation matrix already applied to the context.
     */
    canvasPaintSelf: function( wrapper, matrix ) {
      // TODO: see https://github.com/phetsims/scenery/issues/308
      assert && assert( 'unimplemented: canvasPaintSelf in WebGLNode' );
    },

    /**
     * Renders this Node only (its self) into the Canvas wrapper, in its local coordinate frame.
     * @public
     * @override
     *
     * @param {CanvasContextWrapper} wrapper
     * @param {Matrix3} matrix - The current transformation matrix associated with the wrapper
     */
    renderToCanvasSelf: function( wrapper, matrix ) {
      var width = wrapper.canvas.width;
      var height = wrapper.canvas.height;

      // TODO: Can we reuse the same Canvas? That might save some context creations?
      var scratchCanvas = document.createElement( 'canvas' );
      scratchCanvas.width = width;
      scratchCanvas.height = height;
      var contextOptions = {
        antialias: true,
        preserveDrawingBuffer: true // so we can get the data and render it to the Canvas
      };
      var gl = scratchCanvas.getContext( 'webgl', contextOptions ) || scratchCanvas.getContext( 'experimental-webgl', contextOptions );
      Util.applyWebGLContextDefaults( gl ); // blending, etc.

      var projectionMatrix = new Matrix3().rowMajor(
        2 / width, 0, -1,
        0, -2 / height, 1,
        0, 0, 1 );
      gl.viewport( 0, 0, width, height );

      var PainterType = this.painterType;
      var painter = new PainterType( gl, this );

      painter.paint( matrix, projectionMatrix );
      painter.dispose();

      projectionMatrix.freeToPool();

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
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {WebGLSelfDrawable}
     */
    createWebGLDrawable: function( renderer, instance ) {
      return WebGLNodeDrawable.createFromPool( renderer, instance );
    }
  }, {
    /**
     * @public {number} - Return code from painter.paint() when nothing was painted to the WebGL context.
     */
    PAINTED_NOTHING: 0,

    /**
     * @public {number} - Return code from painter.paint() when something was painted to the WebGL context.
     */
    PAINTED_SOMETHING: 1
  } );

  return WebGLNode;
} );
