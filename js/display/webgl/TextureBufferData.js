//  Copyright 2002-2014, University of Colorado Boulder

/**
 * This WebGL renderer is used to draw images as textures on rectangles.
 * TODO: Can this same pattern be used for interleaved texture coordinates? (Or other interleaved data?)
 * TODO: Work in progress, much to be done here!
 * TODO: Add this file to the list of scenery files (for jshint, etc.)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );

  /**
   *
   * @constructor
   */
  function TextureBufferData() {
    this.vertexArray = [];
  }

  return inherit( Object, TextureBufferData, {
    createFromImageNode: function( imageNode, z,frameRange ) {
      return this.createFromImage( imageNode.x, imageNode.y, z,
        imageNode._image.width, imageNode._image.height, imageNode.image, imageNode.getLocalToGlobalMatrix().toMatrix4(),frameRange );
    },
    createFromImage: function( x, y, z, width, height, image, matrix4,frameRange ) {
      //TODO: Check to see if any of the sprite sheets already contains that image
      //TODO: If none of the sprite sheets contained that image, then mark the spritesheet as dirty
      //TODO: and send it to the GPU after updating

      var textureBufferData = this;
      var index = this.vertexArray.length;

      var x1 = x;
      var x2 = x + width;
      var y1 = y;
      var y2 = y + height;

      var u1 = frameRange.bounds.minX;
      var u2 = frameRange.bounds.maxX;
      var v1 = frameRange.bounds.minY;
      var v2 = frameRange.bounds.maxY;

      this.vertexArray.push(
        x1, y1, z, u1, v1, matrix4.m00(), matrix4.m01(), matrix4.m02(), matrix4.m10(), matrix4.m11(), matrix4.m12(),
        x2, y1, z, u2, v1, matrix4.m00(), matrix4.m01(), matrix4.m02(), matrix4.m10(), matrix4.m11(), matrix4.m12(),
        x1, y2, z, u1, v2, matrix4.m00(), matrix4.m01(), matrix4.m02(), matrix4.m10(), matrix4.m11(), matrix4.m12(),
        x1, y2, z, u1, v2, matrix4.m00(), matrix4.m01(), matrix4.m02(), matrix4.m10(), matrix4.m11(), matrix4.m12(),
        x2, y1, z, u2, v1, matrix4.m00(), matrix4.m01(), matrix4.m02(), matrix4.m10(), matrix4.m11(), matrix4.m12(),
        x2, y2, z, u2, v2, matrix4.m00(), matrix4.m01(), matrix4.m02(), matrix4.m10(), matrix4.m11(), matrix4.m12()
      );

      //Track the index so it can delete itself, update itself, etc.
      //TODO: Move to a separate class.
      return {
        startIndex: index,
        endIndex: textureBufferData.vertexArray.length,
        image: image,
        frameRange:frameRange,
        setTransform: function( matrix4 ) {
          for ( var i = 0; i < 6; i++ ) {
            textureBufferData.vertexArray[ index + 5 + i * 11 ] = matrix4.m00();
            textureBufferData.vertexArray[ index + 6 + i * 11 ] = matrix4.m01();
            textureBufferData.vertexArray[ index + 7 + i * 11 ] = matrix4.m03();
            textureBufferData.vertexArray[ index + 8 + i * 11 ] = matrix4.m10();
            textureBufferData.vertexArray[ index + 9 + i * 11 ] = matrix4.m11();
            textureBufferData.vertexArray[ index + 10 + i * 11 ] = matrix4.m13();
          }
        },
        setXWidth: function( x, width ) {
          textureBufferData.vertexArray[ index ] = x;
          textureBufferData.vertexArray[ index + 2 ] = x + width;
          textureBufferData.vertexArray[ index + 4 ] = x;
          textureBufferData.vertexArray[ index + 6 ] = x + width;
          textureBufferData.vertexArray[ index + 8 ] = x + width;
          textureBufferData.vertexArray[ index + 10 ] = x;
        },
        setRect: function( x, y, width, height ) {

          textureBufferData.vertexArray[ index ] = x;
          textureBufferData.vertexArray[ index + 1 ] = y;

          textureBufferData.vertexArray[ index + 2 ] = x + width;
          textureBufferData.vertexArray[ index + 3 ] = y;

          textureBufferData.vertexArray[ index + 4 ] = x;
          textureBufferData.vertexArray[ index + 5 ] = y + height;

          textureBufferData.vertexArray[ index + 6 ] = x + width;
          textureBufferData.vertexArray[ index + 7 ] = y + height;

          textureBufferData.vertexArray[ index + 8 ] = x + width;
          textureBufferData.vertexArray[ index + 9 ] = y;

          textureBufferData.vertexArray[ index + 10 ] = x;
          textureBufferData.vertexArray[ index + 11 ] = y + height;
        }
      };
    }
  } );
} );