//  Copyright 2002-2014, University of Colorado Boulder

/**
 * This renderer shows WebGL textures.  To achieve performance goals, it is important to minimize the number of draw calls.
 * So we try to add as many sprites into each texture as possible, and render all with a small number of draw calls.
 *
 * Adds as many images into a single SpriteSheet using a bin packing algorithm.A new Sprite will be created If a single
 * SpriteSheet cannot accommodate them. If there are 3  SpriteSheets, there would be 3 TriangleBuggerData, 3 VertexBuffers,3 Textures
 * and 3 draw calls.
 * Before each draw call, the appropriate Texture will be activated and the drawTriangles will pick the right TriangleBufferData
 * SpriteSheetIndex property of FrameRange is used to associate textures,TriangleBuggerData and VertexBuffers.
 *
 * Too many distinct Images with bigger dimensions will result in more than one SpriteSheet.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var TextureBufferData = require( 'SCENERY/display/webgl/TextureBufferData' );
  var SpriteSheetCollection = require( 'SCENERY/display/webgl/SpriteSheetCollection' );

  // shaders
  var colorVertexShader = require( 'text!SCENERY/display/webgl/texture.vert' );
  var colorFragmentShader = require( 'text!SCENERY/display/webgl/texture.frag' );

  /**
   * @constructor
   */
  function TextureRenderer( gl, backingScale, canvas ) {
    this.gl = gl;
    this.canvas = canvas;
    this.backingScale = backingScale;

    var toShader = function( source, type, typeString ) {
      var shader = gl.createShader( type );
      gl.shaderSource( shader, source );
      gl.compileShader( shader );
      if ( !gl.getShaderParameter( shader, gl.COMPILE_STATUS ) ) {
        console.log( 'ERROR IN ' + typeString + ' SHADER : ' + gl.getShaderInfoLog( shader ) );
        return false;
      }
      return shader;
    };

    this.colorShaderProgram = gl.createProgram();
    var program = this.colorShaderProgram;
    gl.attachShader( this.colorShaderProgram, toShader( colorVertexShader, gl.VERTEX_SHADER, 'VERTEX' ) );
    gl.attachShader( this.colorShaderProgram, toShader( colorFragmentShader, gl.FRAGMENT_SHADER, 'FRAGMENT' ) );
    gl.linkProgram( this.colorShaderProgram );

    // look up where the vertex data needs to go.
    this.positionLocation = gl.getAttribLocation( program, 'aPosition' );
    this.texCoordLocation = gl.getAttribLocation( program, 'aTextureCoordinate' );
    this.transform1AttributeLocation = gl.getAttribLocation( this.colorShaderProgram, 'aTransform1' );
    this.transform2AttributeLocation = gl.getAttribLocation( this.colorShaderProgram, 'aTransform2' );

    // lookup uniforms
    this.resolutionLocation = gl.getUniformLocation( program, 'uResolution' );
    gl.blendFunc( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA );
    gl.enable( this.gl.BLEND );

    this.spriteSheetCollection = new SpriteSheetCollection();
    // TODO: Compare this same idea to triangle strips
    //Each textureBufferData manages the indices within a single array, so that disjoint geometries can be represented easily here.
    this.textureBufferDataArray = [];
    // Create a buffer for the position of the rectangle corners.
    this.vertexBufferArray = [];
    this.textureArray = [];
    // List of Vertex Array to Keep for updating sublist
    this.vertexArrayList = [];
  }

  return inherit( Object, TextureRenderer, {
    createFromImageNode: function( imageNode, z ) {
      var frameRange = this.spriteSheetCollection.addImage( imageNode.image );
      if ( !this.textureBufferDataArray[ frameRange.spriteSheetIndex ] ) {
        // if there is no textureBuffer,VertextBuffer,textures entry  for this SpriteSheet so create one
        this.textureBufferDataArray[ frameRange.spriteSheetIndex ] = new TextureBufferData();
        this.vertexBufferArray[ frameRange.spriteSheetIndex ] = this.gl.createBuffer();
        this.textureArray[ frameRange.spriteSheetIndex ] = this.gl.createTexture();
      }
      var textureBufferData = this.textureBufferDataArray[ frameRange.spriteSheetIndex ];
      return textureBufferData.createFromImage( 0, 0, z,
        imageNode._image.width, imageNode._image.height, imageNode.image, imageNode.getLocalToGlobalMatrix().toMatrix4(), frameRange );
    },

    draw: function() {
      for ( var i = 0; i < this.textureArray.length; i++ ) {
        this.doDraw( i );
      }
    },

    /**
     * @private
     */
    doDraw: function( activeTextureIndex ) {
      var gl = this.gl;

      gl.useProgram( this.colorShaderProgram );
      gl.enableVertexAttribArray( this.positionLocation );
      gl.enableVertexAttribArray( this.texCoordLocation );
      gl.enableVertexAttribArray( this.transform1AttributeLocation );
      gl.enableVertexAttribArray( this.transform2AttributeLocation );

      // bind and activate the correct texture
      // TODO: Does this need to be done every frame?
      gl.bindTexture( gl.TEXTURE_2D, this.textureArray[ activeTextureIndex ] );
      // Activate the correct texture
      gl.activeTexture( gl.TEXTURE0 + activeTextureIndex );

      // set the resolution
      // TODO: This backing scale multiply seems very buggy and contradicts everything we know!
      // TODO: Does this need to be done every frame?
      // Still, it gives the right behavior on iPad3 and OSX (non-retina).  Should be discussed and investigated.
      gl.uniform2f( this.resolutionLocation, this.canvas.width / this.backingScale, this.canvas.height / this.backingScale );

      var step = Float32Array.BYTES_PER_ELEMENT;
      var total = 3 + 2 + 3 + 3;
      var stride = step * total;

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBufferArray[ activeTextureIndex ] );
      gl.vertexAttribPointer( this.positionLocation, 3, gl.FLOAT, false, stride, 0 );
      gl.vertexAttribPointer( this.texCoordLocation, 2, gl.FLOAT, false, stride, step * 3 );
      gl.vertexAttribPointer( this.transform1AttributeLocation, 3, gl.FLOAT, false, stride, step * (3 + 2) );
      gl.vertexAttribPointer( this.transform2AttributeLocation, 3, gl.FLOAT, false, stride, step * (3 + 2 + 3) );

      // Draw the rectangle.
      gl.drawArrays( gl.TRIANGLES, 0, this.textureBufferDataArray[ activeTextureIndex ].vertexArray.length / total );

      gl.disableVertexAttribArray( this.texCoordLocation );
      gl.disableVertexAttribArray( this.positionLocation );
      gl.disableVertexAttribArray( this.transform1AttributeLocation );
      gl.disableVertexAttribArray( this.transform2AttributeLocation );

      gl.bindTexture( gl.TEXTURE_2D, null );
    },

    getSpriteSheets: function() {
      return this.spriteSheetCollection.spriteSheets;
    },

    /**
     * Iterate through all of the sprite sheets and register the dirty ones with the GPU as texture units.
     */
    bindDirtyTextures: function() {
      var gl = this.gl;
      var spriteSheets = this.getSpriteSheets();
      for ( var i = 0; i < spriteSheets.length; i++ ) {
        var spriteSheet = spriteSheets[ i ];
        if ( spriteSheet.dirty ) {
          gl.bindTexture( gl.TEXTURE_2D, this.textureArray[ i ] );
          gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
          gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );
          gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR );
          gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR );
          gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, spriteSheet.image );
          gl.generateMipmap( gl.TEXTURE_2D );

          gl.bindTexture( gl.TEXTURE_2D, null );

          spriteSheet.dirty = false;
        }
      }
    },

    bindVertexBuffer: function() {
      var gl = this.gl;
      var spriteSheets = this.getSpriteSheets();
      for ( var i = 0; i < spriteSheets.length; i++ ) {
        gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBufferArray[ i ] );
        // Keep track of the vertexArray for updating sublists of it
        this.vertexArrayList[ i ] = new Float32Array( this.textureBufferDataArray[ i ].vertexArray );
        gl.bufferData( gl.ARRAY_BUFFER, this.vertexArrayList[ i ], gl.DYNAMIC_DRAW );
      }

    },

    updateTriangleBuffer: function( geometry ) {
      var gl = this.gl;

      var frameRange = geometry.frameRange;
      // Update the vertex locations
      // Use a buffer view to only update the changed vertices
      // like //see http://stackoverflow.com/questions/19892022/webgl-optimizing-a-vertex-buffer-that-changes-values-vertex-count-every-frame
      // See also http://stackoverflow.com/questions/5497722/how-can-i-animate-an-object-in-webgl-modify-specific-vertices-not-full-transfor
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBufferArray[ frameRange.spriteSheetIndex ] );

      var vertexArray = this.vertexArrayList[ frameRange.spriteSheetIndex ];
      //Update the Float32Array values
      for ( var i = geometry.startIndex; i < geometry.endIndex; i++ ) {
        vertexArray[ i ] = this.textureBufferDataArray[ frameRange.spriteSheetIndex ].vertexArray[ i ];
      }
      // Isolate the subarray of changed values
      var subArray = vertexArray.subarray( geometry.startIndex, geometry.endIndex );

      // Send new values to the GPU
      // See https://www.khronos.org/webgl/public-mailing-list/archives/1201/msg00110.html
      // The the offset is the index times the bytes per value
      gl.bufferSubData( gl.ARRAY_BUFFER, geometry.startIndex * 4, subArray );
    }

  } );
} );