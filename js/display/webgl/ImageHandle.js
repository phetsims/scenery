//  Copyright 2002-2014, University of Colorado Boulder

/**
 *
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var TextureBufferData = require( 'SCENERY/display/webgl/TextureBufferData' );

  /**
   *
   * @constructor
   */
  function ImageHandle( textureRenderer, imageNode, z ) {

    var frameRange = textureRenderer.spriteSheetCollection.addImage( imageNode.image );

    // If there is no textureBuffer/VertexBuffer/textures entry for this SpriteSheet so create one
    if ( !textureRenderer.textureBufferDataArray[ frameRange.spriteSheetIndex ] ) {

      textureRenderer.textureBufferDataArray[ frameRange.spriteSheetIndex ] = new TextureBufferData();
      textureRenderer.vertexBufferArray[ frameRange.spriteSheetIndex ] = textureRenderer.gl.createBuffer();
      textureRenderer.textureArray[ frameRange.spriteSheetIndex ] = textureRenderer.gl.createTexture();
    }
    var textureBufferData = textureRenderer.textureBufferDataArray[ frameRange.spriteSheetIndex ];

    var x = 0;
    var y = 0;
    var width = imageNode._image.width;
    var height = imageNode._image.height;
    var image = imageNode.image;
    var matrix4 = imageNode.getLocalToGlobalMatrix().toMatrix4();

    //TODO: Check to see if any of the sprite sheets already contains that image
    //TODO: If none of the sprite sheets contained that image, then mark the spritesheet as dirty
    //TODO: and send it to the GPU after updating

    var index = textureBufferData.vertexArray.length;

    var x1 = x;
    var x2 = x + width;
    var y1 = y;
    var y2 = y + height;

    var u0 = frameRange.bounds.minX;
    var u1 = frameRange.bounds.maxX;
    var v0 = frameRange.bounds.minY;
    var v1 = frameRange.bounds.maxY;

    textureBufferData.vertexArray.push(
      x1, y1, z, u0, v0, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
      x2, y1, z, u1, v0, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
      x1, y2, z, u0, v1, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
      x1, y2, z, u0, v1, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
      x2, y1, z, u1, v0, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
      x2, y2, z, u1, v1, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13()
    );

    this.startIndex = index;
    this.endIndex = textureBufferData.vertexArray.length;
    this.image = image;
    this.frameRange = frameRange;
    this.textureBufferData = textureBufferData;
    //Track the index so it can delete itself, update itself, etc.
    //TODO: Move to a separate class.
  }

  return inherit( Object, ImageHandle, {
    setTransform: function( matrix4 ) {
      var textureBufferData = this.textureBufferData;
      var index = this.startIndex;
      for ( var i = 0; i < 6; i++ ) {
        textureBufferData.vertexArray[ index + 5 + i * 11 ] = matrix4.m00();
        textureBufferData.vertexArray[ index + 6 + i * 11 ] = matrix4.m01();
        textureBufferData.vertexArray[ index + 7 + i * 11 ] = matrix4.m03();
        textureBufferData.vertexArray[ index + 8 + i * 11 ] = matrix4.m10();
        textureBufferData.vertexArray[ index + 9 + i * 11 ] = matrix4.m11();
        textureBufferData.vertexArray[ index + 10 + i * 11 ] = matrix4.m13();
      }
    },
    setRect: function( x, y, width, height ) {
      var index = this.startIndex;
      var textureBufferData = this.textureBufferData;
      var x1 = x;
      var y1 = y;
      var x2 = x1 + width;
      var y2 = y1 + height;
      textureBufferData.vertexArray[ index + 0 ] = x1;
      textureBufferData.vertexArray[ index + 1 ] = y1;

      textureBufferData.vertexArray[ index + 11 ] = x2;
      textureBufferData.vertexArray[ index + 12 ] = y1;

      textureBufferData.vertexArray[ index + 22 ] = x1;
      textureBufferData.vertexArray[ index + 23 ] = y2;

      textureBufferData.vertexArray[ index + 33 ] = x1;
      textureBufferData.vertexArray[ index + 34 ] = y2;

      textureBufferData.vertexArray[ index + 44 ] = x2;
      textureBufferData.vertexArray[ index + 45 ] = y1;

      textureBufferData.vertexArray[ index + 55 ] = x2;
      textureBufferData.vertexArray[ index + 56 ] = y2;

    }

  } );
} );