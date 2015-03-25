//  Copyright 2002-2014, University of Colorado Boulder

/**
 * ImageHandle is created when using the WebGL renderer, it corresponds to a single scenery Image
 * and coordinates with SpriteSheetCollection to create an (updateable) image texture for rendering.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @autor Sharf Ashraf
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
    this.imageNode = imageNode;
    this.z = z;
    this.textureRenderer = textureRenderer;
    this.frameRange = textureRenderer.spriteSheetCollection.addImage( imageNode.image );

    // If there is no textureBuffer/VertexBuffer/textures entry for this SpriteSheet so create one
    if ( !textureRenderer.textureBufferDataArray[ this.frameRange.spriteSheetIndex ] ) {

      textureRenderer.textureBufferDataArray[ this.frameRange.spriteSheetIndex ] = new TextureBufferData();
      textureRenderer.vertexBufferArray[ this.frameRange.spriteSheetIndex ] = textureRenderer.gl.createBuffer();
      textureRenderer.textureArray[ this.frameRange.spriteSheetIndex ] = textureRenderer.gl.createTexture();
    }
    this.textureBufferData = textureRenderer.textureBufferDataArray[ this.frameRange.spriteSheetIndex ];

    //TODO: Check to see if any of the sprite sheets already contains that image
    //TODO: If none of the sprite sheets contained that image, then mark the spritesheet as dirty
    //TODO: and send it to the GPU after updating

    var range = this.textureBufferData.reserveVertices( 6 );
    this.startIndex = range.startIndex;
    this.endIndex = range.endIndex;

    this.update();
  }

  return inherit( Object, ImageHandle, {
    update: function() {

      var x = 0;
      var y = 0;
      var z = this.z;

      var imageNode = this.imageNode;

      // Check to see if any of the sprite sheets already contains that image
      this.frameRange = this.textureRenderer.spriteSheetCollection.getFrameRange( imageNode.image );

      if ( !this.frameRange ) {
        this.frameRange = this.textureRenderer.spriteSheetCollection.addImage( imageNode.image );
      }

      var width = imageNode.getImageWidth();
      var height = imageNode.getImageHeight();
      var matrix4 = imageNode.getLocalToGlobalMatrix().toAffineMatrix4();

      var x1 = x;
      var x2 = x + width;
      var y1 = y;
      var y2 = y + height;

      var u0 = this.frameRange.bounds.minX;
      var u1 = this.frameRange.bounds.maxX;
      var v0 = this.frameRange.bounds.minY;
      var v1 = this.frameRange.bounds.maxY;

      //Track the index so it can delete itself, update itself, etc.
      var newElements = [
        x1, y1, z, u0, v0, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
        x2, y1, z, u1, v0, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
        x1, y2, z, u0, v1, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
        x1, y2, z, u0, v1, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
        x2, y1, z, u1, v0, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13(),
        x2, y2, z, u1, v1, matrix4.m00(), matrix4.m01(), matrix4.m03(), matrix4.m10(), matrix4.m11(), matrix4.m13()
      ];
      for ( var i = 0; i < newElements.length; i++ ) {
        this.textureBufferData.vertexArray[ this.startIndex + i ] = newElements[ i ];
      }
      this.textureRenderer.updateTriangleBuffer( this );
    }
  } );
} );