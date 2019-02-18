// Copyright 2016, University of Colorado Boulder

/**
 * WebGL drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var ImageStatefulDrawable = require( 'SCENERY/display/drawables/ImageStatefulDrawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var Vector2 = require( 'DOT/Vector2' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );

  // For alignment, we keep things to 8 components, aligned on 4-byte boundaries.
  // See https://developer.apple.com/library/ios/documentation/3DDrawing/Conceptual/OpenGLES_ProgrammingGuide/TechniquesforWorkingwithVertexData/TechniquesforWorkingwithVertexData.html#//apple_ref/doc/uid/TP40008793-CH107-SW15
  var WEBGL_COMPONENTS = 5; // format [X Y U V A] for 6 vertices

  var VERTEX_0_OFFSET = WEBGL_COMPONENTS * 0;
  var VERTEX_1_OFFSET = WEBGL_COMPONENTS * 1;
  var VERTEX_2_OFFSET = WEBGL_COMPONENTS * 2;
  var VERTEX_3_OFFSET = WEBGL_COMPONENTS * 3;
  var VERTEX_4_OFFSET = WEBGL_COMPONENTS * 4;
  var VERTEX_5_OFFSET = WEBGL_COMPONENTS * 5;

  var VERTEX_X_OFFSET = 0;
  var VERTEX_Y_OFFSET = 1;
  var VERTEX_U_OFFSET = 2;
  var VERTEX_V_OFFSET = 3;
  var VERTEX_A_OFFSET = 4;

  /**
   * A generated WebGLSelfDrawable whose purpose will be drawing our Image. One of these drawables will be created
   * for each displayed instance of an Image.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  function ImageWebGLDrawable( renderer, instance ) {
    this.initializeWebGLSelfDrawable( renderer, instance );

    if ( !this.vertexArray ) {
      // for 6 vertices
      this.vertexArray = new Float32Array( WEBGL_COMPONENTS * 6 ); // 5-length components for 6 vertices (2 tris).
    }

    // corner vertices in the relative transform root coordinate space
    this.upperLeft = new Vector2( 0, 0 );
    this.lowerLeft = new Vector2( 0, 0 );
    this.upperRight = new Vector2( 0, 0 );
    this.lowerRight = new Vector2( 0, 0 );

    this.xyDirty = true; // is our vertex position information out of date?
    this.uvDirty = true; // is our UV information out of date?
    this.updatedOnce = false;

    // {SpriteSheet.Sprite} exported for WebGLBlock's rendering loop
    this.sprite = null;
  }

  scenery.register( 'ImageWebGLDrawable', ImageWebGLDrawable );

  inherit( WebGLSelfDrawable, ImageWebGLDrawable, {
    // TODO: doc
    webglRenderer: Renderer.webglTexturedTriangles,

    onAddToBlock: function( webglBlock ) {
      this.webglBlock = webglBlock; // TODO: do we need this reference?
      this.markDirty();

      this.reserveSprite();
    },

    onRemoveFromBlock: function( webglBlock ) {
      this.unreserveSprite();
    },

    reserveSprite: function() {
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
      var width = this.node.getImageWidth();
      var height = this.node.getImageHeight();

      // if we have a width/height, we'll load a sprite
      this.sprite = ( width > 0 && height > 0 ) ? this.webglBlock.addSpriteSheetImage( this.node._image, width, height ) : null;

      // full updates on everything if our sprite changes
      this.xyDirty = true;
      this.uvDirty = true;
    },

    unreserveSprite: function() {
      if ( this.sprite ) {
        this.webglBlock.removeSpriteSheetImage( this.sprite );
      }
      this.sprite = null;
    },

    // @override
    markTransformDirty: function() {
      this.xyDirty = true;

      WebGLSelfDrawable.prototype.markTransformDirty.call( this );
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
      this.xyDirty = true; // vertex positions can depend on image width/height
      this.uvDirty = true;

      this.markDirty();
    },

    /**
     * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
     * @public
     * @override
     *
     * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
     *                      be done).
     */
    update: function() {
      // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
      if ( !WebGLSelfDrawable.prototype.update.call( this ) ) {
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

        var uvBounds = this.sprite.uvBounds;

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

        var width = this.node.getImageWidth();
        var height = this.node.getImageHeight();

        var transformMatrix = this.instance.relativeTransform.matrix; // with compute need, should always be accurate
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
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      // TODO: disposal of buffers?

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    }
  } );
  ImageStatefulDrawable.mixInto( ImageWebGLDrawable );

  Poolable.mixInto( ImageWebGLDrawable );

  return ImageWebGLDrawable;
} );
