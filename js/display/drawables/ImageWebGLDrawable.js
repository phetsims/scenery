// Copyright 2016-2022, University of Colorado Boulder

/**
 * WebGL drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../../dot/js/Vector2.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { ImageStatefulDrawable, Renderer, scenery, WebGLSelfDrawable } from '../../imports.js';

// For alignment, we keep things to 8 components, aligned on 4-byte boundaries.
// See https://developer.apple.com/library/ios/documentation/3DDrawing/Conceptual/OpenGLES_ProgrammingGuide/TechniquesforWorkingwithVertexData/TechniquesforWorkingwithVertexData.html#//apple_ref/doc/uid/TP40008793-CH107-SW15
const WEBGL_COMPONENTS = 5; // format [X Y U V A] for 6 vertices

const VERTEX_0_OFFSET = WEBGL_COMPONENTS * 0;
const VERTEX_1_OFFSET = WEBGL_COMPONENTS * 1;
const VERTEX_2_OFFSET = WEBGL_COMPONENTS * 2;
const VERTEX_3_OFFSET = WEBGL_COMPONENTS * 3;
const VERTEX_4_OFFSET = WEBGL_COMPONENTS * 4;
const VERTEX_5_OFFSET = WEBGL_COMPONENTS * 5;

const VERTEX_X_OFFSET = 0;
const VERTEX_Y_OFFSET = 1;
const VERTEX_U_OFFSET = 2;
const VERTEX_V_OFFSET = 3;
const VERTEX_A_OFFSET = 4;

class ImageWebGLDrawable extends ImageStatefulDrawable( WebGLSelfDrawable ) {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance );

    // @public {Float32Array} - 5-length components for 6 vertices (2 tris), for 6 vertices
    this.vertexArray = this.vertexArray || new Float32Array( WEBGL_COMPONENTS * 6 );

    // @private {Vector2} - corner vertices in the relative transform root coordinate space
    this.upperLeft = new Vector2( 0, 0 );
    this.lowerLeft = new Vector2( 0, 0 );
    this.upperRight = new Vector2( 0, 0 );
    this.lowerRight = new Vector2( 0, 0 );

    // @private {boolean}
    this.xyDirty = true; // is our vertex position information out of date?
    this.uvDirty = true; // is our UV information out of date?
    this.updatedOnce = false;

    // {SpriteSheet.Sprite} exported for WebGLBlock's rendering loop
    this.sprite = null;
  }

  /**
   * @public
   *
   * @param {WebGLBlock} webGLBlock
   */
  onAddToBlock( webglBlock ) {
    this.webglBlock = webglBlock; // TODO: do we need this reference?
    this.markDirty();

    this.reserveSprite();
  }

  /**
   * @public
   *
   * @param {WebGLBlock} webGLBlock
   */
  onRemoveFromBlock( webglBlock ) {
    this.unreserveSprite();
  }

  /**
   * @private
   */
  reserveSprite() {
    if ( this.sprite ) {
      // if we already reserved a sprite for the image, bail out
      if ( this.sprite.image === this.node._image ) {
        return;
      }
      // otherwise we need to ditch our last reservation before reserving a new sprite
      else {
        this.unreserveSprite();
      }
    }

    // if the width/height isn't loaded yet, we can still use the desired value
    const width = this.node.getImageWidth();
    const height = this.node.getImageHeight();

    // if we have a width/height, we'll load a sprite
    this.sprite = ( width > 0 && height > 0 ) ? this.webglBlock.addSpriteSheetImage( this.node._image, width, height ) : null;

    // full updates on everything if our sprite changes
    this.xyDirty = true;
    this.uvDirty = true;
  }

  /**
   * @private
   */
  unreserveSprite() {
    if ( this.sprite ) {
      this.webglBlock.removeSpriteSheetImage( this.sprite );
    }
    this.sprite = null;
  }

  /**
   * @public
   * @override
   */
  markTransformDirty() {
    this.xyDirty = true;

    super.markTransformDirty();
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
    this.xyDirty = true; // vertex positions can depend on image width/height
    this.uvDirty = true;

    this.markDirty();
  }

  /**
   * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
   * @public
   * @override
   *
   * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
   *                      be done).
   */
  update() {
    // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
    if ( !super.update() ) {
      return false;
    }

    // ensure that we have a reserved sprite (part of the spritesheet)
    this.reserveSprite();

    if ( this.dirtyImageOpacity || !this.updatedOnce ) {
      this.vertexArray[ VERTEX_0_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
      this.vertexArray[ VERTEX_1_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
      this.vertexArray[ VERTEX_2_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
      this.vertexArray[ VERTEX_3_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
      this.vertexArray[ VERTEX_4_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
      this.vertexArray[ VERTEX_5_OFFSET + VERTEX_A_OFFSET ] = this.node._imageOpacity;
    }
    this.updatedOnce = true;

    // if we don't have a sprite (we don't have a loaded image yet), just bail
    if ( !this.sprite ) {
      return false;
    }

    if ( this.uvDirty ) {
      this.uvDirty = false;

      const uvBounds = this.sprite.uvBounds;

      // TODO: consider reversal of minY and maxY usage here for vertical inverse

      // first triangle UVs
      this.vertexArray[ VERTEX_0_OFFSET + VERTEX_U_OFFSET ] = uvBounds.minX; // upper left U
      this.vertexArray[ VERTEX_0_OFFSET + VERTEX_V_OFFSET ] = uvBounds.minY; // upper left V
      this.vertexArray[ VERTEX_1_OFFSET + VERTEX_U_OFFSET ] = uvBounds.minX; // lower left U
      this.vertexArray[ VERTEX_1_OFFSET + VERTEX_V_OFFSET ] = uvBounds.maxY; // lower left V
      this.vertexArray[ VERTEX_2_OFFSET + VERTEX_U_OFFSET ] = uvBounds.maxX; // upper right U
      this.vertexArray[ VERTEX_2_OFFSET + VERTEX_V_OFFSET ] = uvBounds.minY; // upper right V

      // second triangle UVs
      this.vertexArray[ VERTEX_3_OFFSET + VERTEX_U_OFFSET ] = uvBounds.maxX; // upper right U
      this.vertexArray[ VERTEX_3_OFFSET + VERTEX_V_OFFSET ] = uvBounds.minY; // upper right V
      this.vertexArray[ VERTEX_4_OFFSET + VERTEX_U_OFFSET ] = uvBounds.minX; // lower left U
      this.vertexArray[ VERTEX_4_OFFSET + VERTEX_V_OFFSET ] = uvBounds.maxY; // lower left V
      this.vertexArray[ VERTEX_5_OFFSET + VERTEX_U_OFFSET ] = uvBounds.maxX; // lower right U
      this.vertexArray[ VERTEX_5_OFFSET + VERTEX_V_OFFSET ] = uvBounds.maxY; // lower right V
    }

    if ( this.xyDirty ) {
      this.xyDirty = false;

      const width = this.node.getImageWidth();
      const height = this.node.getImageHeight();

      const transformMatrix = this.instance.relativeTransform.matrix; // with compute need, should always be accurate
      transformMatrix.multiplyVector2( this.upperLeft.setXY( 0, 0 ) );
      transformMatrix.multiplyVector2( this.lowerLeft.setXY( 0, height ) );
      transformMatrix.multiplyVector2( this.upperRight.setXY( width, 0 ) );
      transformMatrix.multiplyVector2( this.lowerRight.setXY( width, height ) );

      // first triangle XYs
      this.vertexArray[ VERTEX_0_OFFSET + VERTEX_X_OFFSET ] = this.upperLeft.x;
      this.vertexArray[ VERTEX_0_OFFSET + VERTEX_Y_OFFSET ] = this.upperLeft.y;
      this.vertexArray[ VERTEX_1_OFFSET + VERTEX_X_OFFSET ] = this.lowerLeft.x;
      this.vertexArray[ VERTEX_1_OFFSET + VERTEX_Y_OFFSET ] = this.lowerLeft.y;
      this.vertexArray[ VERTEX_2_OFFSET + VERTEX_X_OFFSET ] = this.upperRight.x;
      this.vertexArray[ VERTEX_2_OFFSET + VERTEX_Y_OFFSET ] = this.upperRight.y;

      // second triangle XYs
      this.vertexArray[ VERTEX_3_OFFSET + VERTEX_X_OFFSET ] = this.upperRight.x;
      this.vertexArray[ VERTEX_3_OFFSET + VERTEX_Y_OFFSET ] = this.upperRight.y;
      this.vertexArray[ VERTEX_4_OFFSET + VERTEX_X_OFFSET ] = this.lowerLeft.x;
      this.vertexArray[ VERTEX_4_OFFSET + VERTEX_Y_OFFSET ] = this.lowerLeft.y;
      this.vertexArray[ VERTEX_5_OFFSET + VERTEX_X_OFFSET ] = this.lowerRight.x;
      this.vertexArray[ VERTEX_5_OFFSET + VERTEX_Y_OFFSET ] = this.lowerRight.y;
    }

    return true;
  }
}

// TODO: doc
ImageWebGLDrawable.prototype.webglRenderer = Renderer.webglTexturedTriangles;

scenery.register( 'ImageWebGLDrawable', ImageWebGLDrawable );

Poolable.mixInto( ImageWebGLDrawable );

export default ImageWebGLDrawable;