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
    return textureBufferData.createFromImage( 0, 0, z,
      imageNode._image.width,
      imageNode._image.height,
      imageNode.image,
      imageNode.getLocalToGlobalMatrix().toMatrix4(),
      frameRange );
  }

  return inherit( Object, ImageHandle, {} );
} );