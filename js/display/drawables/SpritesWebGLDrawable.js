// Copyright 2019-2022, University of Colorado Boulder

/**
 * WebGL drawable for Sprites.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import animationFrameTimer from '../../../../axon/js/animationFrameTimer.js';
import Vector2 from '../../../../dot/js/Vector2.js';
import platform from '../../../../phet-core/js/platform.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { Renderer, scenery, ShaderProgram, SpriteSheet, WebGLSelfDrawable } from '../../imports.js';

// constants
const COMPONENTS = 5; // { X Y U V A }
const FLOAT_QUANTITY = COMPONENTS * 6; // 6 vertices

// scratch values - corner vertices in the relative transform root coordinate space
const upperLeft = new Vector2( 0, 0 );
const lowerLeft = new Vector2( 0, 0 );
const upperRight = new Vector2( 0, 0 );
const lowerRight = new Vector2( 0, 0 );

class SpritesWebGLDrawable extends WebGLSelfDrawable {
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
    this.contextChangeListener = this.onWebGLContextChange.bind( this );

    // @private {SpriteSheet}
    this.spriteSheet = new SpriteSheet( true );

    // @private {Object} - Maps {number} SpriteImage.id => {Bounds2} UV bounds
    this.spriteImageUVMap = {};

    // @private {Float32Array}
    this.vertexArray = new Float32Array( 128 * FLOAT_QUANTITY );

    // @private {Float32Array}
    this.transformMatrixArray = new Float32Array( 9 );

    // @private {function}
    this.spriteChangeListener = this.onSpriteChange.bind( this );

    this.node._sprites.forEach( sprite => {
      sprite.imageProperty.lazyLink( this.spriteChangeListener );
      this.addSpriteImage( sprite.imageProperty.value );
    } );

    // @private {boolean} - See https://github.com/phetsims/natural-selection/issues/243
    this.hasDrawn = false;
  }

  /**
   * Adds a SpriteImage to our SpriteSheet.
   * @private
   *
   * @param {SpriteImage} spriteImage
   */
  addSpriteImage( spriteImage ) {
    this.spriteImageUVMap[ spriteImage.id ] = this.spriteSheet.addImage( spriteImage.image, spriteImage.image.width, spriteImage.image.height ).uvBounds;
  }

  /**
   * Removes a SpriteImage from our SpriteSheet.
   * @private
   *
   * @param {SpriteImage} spriteImage
   */
  removeSpriteImage( spriteImage ) {
    this.spriteSheet.removeImage( spriteImage.image );

    delete this.spriteImageUVMap[ spriteImage.id ];
  }

  /**
   * Called when a Sprite's SpriteImage changes.
   * @private
   *
   * @param {SpriteImage} newSpriteImage
   * @param {SpriteImage} oldSpriteImage
   */
  onSpriteChange( newSpriteImage, oldSpriteImage ) {
    this.removeSpriteImage( oldSpriteImage );
    this.addSpriteImage( newSpriteImage );
  }

  /**
   * Sets up everything with a new WebGL context
   *
   * @private
   */
  setup() {
    const gl = this.webGLBlock.gl;

    this.spriteSheet.initializeContext( gl );

    // @private {ShaderProgram}
    this.shaderProgram = new ShaderProgram( gl, [
      // vertex shader
      'attribute vec2 aVertex;',
      'attribute vec2 aTextureCoord;',
      'attribute float aAlpha;',
      'varying vec2 vTextureCoord;',
      'varying float vAlpha;',
      'uniform mat3 uProjectionMatrix;',
      'uniform mat3 uTransformMatrix;',

      'void main() {',
      '  vTextureCoord = aTextureCoord;',
      '  vAlpha = aAlpha;',
      '  vec3 ndc = uProjectionMatrix * ( uTransformMatrix * vec3( aVertex, 1.0 ) );', // homogeneous map to to normalized device coordinates
      '  gl_Position = vec4( ndc.xy, 0.0, 1.0 );',
      '}'
    ].join( '\n' ), [
      // fragment shader
      'precision mediump float;',
      'varying vec2 vTextureCoord;',
      'varying float vAlpha;',
      'uniform sampler2D uTexture;',

      'void main() {',
      '  vec4 color = texture2D( uTexture, vTextureCoord, -0.7 );', // mipmap LOD bias of -0.7 (for now)
      '  color.a *= vAlpha;',
      '  gl_FragColor = color;', // don't premultiply alpha (we are loading the textures as premultiplied already)
      '}'
    ].join( '\n' ), {
      attributes: [ 'aVertex', 'aTextureCoord', 'aAlpha' ],
      uniforms: [ 'uTexture', 'uProjectionMatrix', 'uTransformMatrix' ]
    } );

    // @private {WebGLBuffer}
    this.vertexBuffer = gl.createBuffer();

    gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
    gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW ); // fully buffer at the start
  }

  /**
   * Callback for when the WebGL context changes. We'll reconstruct the painter.
   * @public
   */
  onWebGLContextChange() {
    this.setup();
  }

  /**
   * Called when this drawable is added to a block.
   * @public
   *
   * @param {WebGLBlock} webGLBlock
   */
  onAddToBlock( webGLBlock ) {
    // @private {WebGLBlock}
    this.webGLBlock = webGLBlock;

    this.setup();

    webGLBlock.glChangedEmitter.addListener( this.contextChangeListener );
  }

  /**
   * Called when this drawable is removed from a block.
   * @public
   *
   * @param {WebGLBlock} webGLBlock
   */
  onRemoveFromBlock( webGLBlock ) {
    webGLBlock.glChangedEmitter.removeListener( this.contextChangeListener );
  }

  /**
   * Draws the WebGL content.
   * @public
   */
  draw() {
    const length = this.node._spriteInstances.length;

    // Don't render anything if we have nothing
    if ( length === 0 ) {
      return 0;
    }

    this.spriteSheet.updateTexture();

    this.shaderProgram.use();

    let vertexArrayIndex = 0;
    let changedLength = false;

    // if our vertex data won't fit, keep doubling the size until it fits
    while ( FLOAT_QUANTITY * length > this.vertexArray.length ) {
      this.vertexArray = new Float32Array( this.vertexArray.length * 2 );
      changedLength = true;
    }

    for ( let i = 0; i < length; i++ ) {
      const spriteInstance = this.node._spriteInstances[ i ];
      const spriteImage = spriteInstance.sprite.imageProperty.value;
      const alpha = spriteInstance.alpha * spriteImage.imageOpacity;
      const uvBounds = this.spriteImageUVMap[ spriteImage.id ];
      const matrix = spriteInstance.matrix;
      const offset = spriteImage.offset;

      const width = spriteImage.image.width;
      const height = spriteImage.image.height;

      // Compute our vertices
      matrix.multiplyVector2( upperLeft.setXY( -offset.x, -offset.y ) );
      matrix.multiplyVector2( lowerLeft.setXY( -offset.x, height - offset.y ) );
      matrix.multiplyVector2( upperRight.setXY( width - offset.x, -offset.y ) );
      matrix.multiplyVector2( lowerRight.setXY( width - offset.x, height - offset.y ) );

      // copy our vertex data into the main array (consensus was that this is the fastest way to fill in data)
      this.vertexArray[ vertexArrayIndex + 0 ] = upperLeft.x;
      this.vertexArray[ vertexArrayIndex + 1 ] = upperLeft.y;
      this.vertexArray[ vertexArrayIndex + 2 ] = uvBounds.minX;
      this.vertexArray[ vertexArrayIndex + 3 ] = uvBounds.minY;
      this.vertexArray[ vertexArrayIndex + 4 ] = alpha;
      this.vertexArray[ vertexArrayIndex + 5 ] = lowerLeft.x;
      this.vertexArray[ vertexArrayIndex + 6 ] = lowerLeft.y;
      this.vertexArray[ vertexArrayIndex + 7 ] = uvBounds.minX;
      this.vertexArray[ vertexArrayIndex + 8 ] = uvBounds.maxY;
      this.vertexArray[ vertexArrayIndex + 9 ] = alpha;
      this.vertexArray[ vertexArrayIndex + 10 ] = upperRight.x;
      this.vertexArray[ vertexArrayIndex + 11 ] = upperRight.y;
      this.vertexArray[ vertexArrayIndex + 12 ] = uvBounds.maxX;
      this.vertexArray[ vertexArrayIndex + 13 ] = uvBounds.minY;
      this.vertexArray[ vertexArrayIndex + 14 ] = alpha;
      this.vertexArray[ vertexArrayIndex + 15 ] = upperRight.x;
      this.vertexArray[ vertexArrayIndex + 16 ] = upperRight.y;
      this.vertexArray[ vertexArrayIndex + 17 ] = uvBounds.maxX;
      this.vertexArray[ vertexArrayIndex + 18 ] = uvBounds.minY;
      this.vertexArray[ vertexArrayIndex + 19 ] = alpha;
      this.vertexArray[ vertexArrayIndex + 20 ] = lowerLeft.x;
      this.vertexArray[ vertexArrayIndex + 21 ] = lowerLeft.y;
      this.vertexArray[ vertexArrayIndex + 22 ] = uvBounds.minX;
      this.vertexArray[ vertexArrayIndex + 23 ] = uvBounds.maxY;
      this.vertexArray[ vertexArrayIndex + 24 ] = alpha;
      this.vertexArray[ vertexArrayIndex + 25 ] = lowerRight.x;
      this.vertexArray[ vertexArrayIndex + 26 ] = lowerRight.y;
      this.vertexArray[ vertexArrayIndex + 27 ] = uvBounds.maxX;
      this.vertexArray[ vertexArrayIndex + 28 ] = uvBounds.maxY;
      this.vertexArray[ vertexArrayIndex + 29 ] = alpha;

      vertexArrayIndex += FLOAT_QUANTITY;
    }

    const gl = this.webGLBlock.gl;

    // (uniform) projection transform into normalized device coordinates
    gl.uniformMatrix3fv( this.shaderProgram.uniformLocations.uProjectionMatrix, false, this.webGLBlock.projectionMatrixArray );

    // (uniform) transformation matrix that is common to all sprites
    this.instance.relativeTransform.matrix.copyToArray( this.transformMatrixArray );
    gl.uniformMatrix3fv( this.shaderProgram.uniformLocations.uTransformMatrix, false, this.transformMatrixArray );

    gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
    // if we increased in length, we need to do a full bufferData to resize it on the GPU side
    if ( changedLength ) {
      gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW ); // fully buffer at the start
    }
    // otherwise do a more efficient update that only sends part of the array over
    else {
      gl.bufferSubData( gl.ARRAY_BUFFER, 0, this.vertexArray.subarray( 0, vertexArrayIndex ) );
    }

    const sizeOfFloat = Float32Array.BYTES_PER_ELEMENT;
    const stride = COMPONENTS * sizeOfFloat;
    gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aVertex, 2, gl.FLOAT, false, stride, 0 * sizeOfFloat );
    gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aTextureCoord, 2, gl.FLOAT, false, stride, 2 * sizeOfFloat );
    gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aAlpha, 1, gl.FLOAT, false, stride, 4 * sizeOfFloat );

    gl.activeTexture( gl.TEXTURE0 );
    gl.bindTexture( gl.TEXTURE_2D, this.spriteSheet.texture );
    gl.uniform1i( this.shaderProgram.uniformLocations.uTexture, 0 );

    gl.drawArrays( gl.TRIANGLES, 0, vertexArrayIndex / COMPONENTS );

    gl.bindTexture( gl.TEXTURE_2D, null );

    this.shaderProgram.unuse();

    // See https://github.com/phetsims/natural-selection/issues/243
    if ( !this.hasDrawn && platform.safari ) {
      // Redraw once more if we're in Safari, since it's undetermined why an initial draw isn't working.
      // Everything seems to otherwise be in place.
      animationFrameTimer.setTimeout( () => this.markDirty(), 0 );
    }
    this.hasDrawn = true;

    return 1;
  }

  /**
   * Disposes the drawable.
   * @public
   * @override
   */
  dispose() {
    this.node._sprites.forEach( sprite => {
      sprite.imageProperty.unlink( this.spriteChangeListener );
    } );

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
}

// We use a custom renderer for the needed flexibility
SpritesWebGLDrawable.prototype.webglRenderer = Renderer.webglCustom;

scenery.register( 'SpritesWebGLDrawable', SpritesWebGLDrawable );

Poolable.mixInto( SpritesWebGLDrawable );

export default SpritesWebGLDrawable;