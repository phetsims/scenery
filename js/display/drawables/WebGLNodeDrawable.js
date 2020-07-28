// Copyright 2016-2020, University of Colorado Boulder

/**
 * WebGL drawable for WebGLNode.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import inherit from '../../../../phet-core/js/inherit.js';
import WebGLNode from '../../nodes/WebGLNode.js';
import scenery from '../../scenery.js';
import Renderer from '../Renderer.js';
import WebGLSelfDrawable from '../WebGLSelfDrawable.js';

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
  // We use a custom renderer for the needed flexibility
  webglRenderer: Renderer.webglCustom,

  /**
   * Creates an instance of our Node's "painter" type.
   * @private
   *
   * @returns {*} - Whatever node.painterType is will be the type.
   */
  createPainter: function() {
    const PainterType = this.node.painterType;
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
    const matrix = this.instance.relativeTransform.matrix;

    const painted = this.painter.paint( matrix, this.webGLBlock.projectionMatrix );

    assert && assert( painted === WebGLNode.PAINTED_SOMETHING || painted === WebGLNode.PAINTED_NOTHING );
    assert && assert( WebGLNode.PAINTED_NOTHING === 0 && WebGLNode.PAINTED_SOMETHING === 1,
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

export default WebGLNodeDrawable;