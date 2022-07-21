// Copyright 2016-2022, University of Colorado Boulder

/**
 * WebGL drawable for WebGLNode.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { Renderer, scenery, WebGLNode, WebGLSelfDrawable } from '../../imports.js';

class WebGLNodeDrawable extends WebGLSelfDrawable {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance );

    // @private {function}
    this.contextChangeListener = this.contextChangeListener || this.onWebGLContextChange.bind( this );

    // @private {*} - Will be set to whatever type node.painterType is.
    this.painter = null;
  }

  /**
   * Creates an instance of our Node's "painter" type.
   * @private
   *
   * @returns {*} - Whatever node.painterType is will be the type.
   */
  createPainter() {
    const PainterType = this.node.painterType;
    return new PainterType( this.webGLBlock.gl, this.node );
  }

  /**
   * Callback for when the WebGL context changes. We'll reconstruct the painter.
   * @public
   */
  onWebGLContextChange() {
    //TODO: Should a function be added for "disposeNonWebGL"?

    // Create the new painter
    this.painter = this.createPainter();
  }

  /**
   * @public
   *
   * @param {WebGLBlock} webGLBlock
   */
  onAddToBlock( webGLBlock ) {
    // @private {WebGLBlock}
    this.webGLBlock = webGLBlock;

    this.painter = this.createPainter();

    webGLBlock.glChangedEmitter.addListener( this.contextChangeListener );
  }

  /**
   * @public
   *
   * @param {WebGLBlock} webGLBlock
   */
  onRemoveFromBlock( webGLBlock ) {
    webGLBlock.glChangedEmitter.removeListener( this.contextChangeListener );
  }

  /**
   * @public
   *
   * @returns {WebGLNode.PAINTED_NOTHING|WebGLNode.PAINTED_SOMETHING}
   */
  draw() {
    // we have a precompute need
    const matrix = this.instance.relativeTransform.matrix;

    const painted = this.painter.paint( matrix, this.webGLBlock.projectionMatrix );

    assert && assert( painted === WebGLNode.PAINTED_SOMETHING || painted === WebGLNode.PAINTED_NOTHING );
    assert && assert( WebGLNode.PAINTED_NOTHING === 0 && WebGLNode.PAINTED_SOMETHING === 1,
      'Ensure we can pass the value through directly to indicate whether draw calls were made' );

    return painted;
  }

  /**
   * Disposes the drawable.
   * @public
   * @override
   */
  dispose() {
    this.painter.dispose();
    this.painter = null;

    if ( this.webGLBlock ) {
      this.webGLBlock = null;
    }

    // super
    super.dispose();
  }

  /**
   * A "catch-all" dirty method that directly marks the paintDirty flag and triggers propagation of dirty
   * information. This can be used by other mark* methods, or directly itself if the paintDirty flag is checked.
   * @public
   *
   * It should be fired (indirectly or directly) for anything besides transforms that needs to make a drawable
   * dirty.
   */
  markPaintDirty() {
    this.markDirty();
  }

  // forward call to the WebGLNode
  get shaderAttributes() {
    return this.node.shaderAttributes;
  }
}

// We use a custom renderer for the needed flexibility
WebGLNodeDrawable.prototype.webglRenderer = Renderer.webglCustom;

scenery.register( 'WebGLNodeDrawable', WebGLNodeDrawable );

Poolable.mixInto( WebGLNodeDrawable );

export default WebGLNodeDrawable;