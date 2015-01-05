//  Copyright 2002-2014, University of Colorado Boulder

/**
 * This renderer shows WebGL textures.  To achieve performance goals, it is important to minimize the number of draw calls.
 * So we try to add as many sprites into each texture as possible, and render all with a small number of draw calls.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var TextureBufferData = require( 'SCENERY/display/webgl/TextureBufferData' );

  // shaders
  var colorVertexShader = require( 'text!SCENERY/display/webgl/texture.vert' );
  var colorFragmentShader = require( 'text!SCENERY/display/webgl/texture.frag' );

  /**
   *
   * @constructor
   */
  function TextureRenderer( gl, backingScale, canvas ) {
    var textureRenderer = this;
    this.gl = gl;
    this.canvas = canvas;
    this.backingScale = backingScale;

    // Manages the indices within a single array, so that disjoint geometries can be represented easily here.
    // TODO: Compare this same idea to triangle strips
    this.textureBufferData = new TextureBufferData();

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

    // Create a texture.
    this.texture = gl.createTexture();

    // lookup uniforms
    this.resolutionLocation = gl.getUniformLocation( program, 'uResolution' );

    // Create a buffer for the position of the rectangle corners.
    this.vertexBuffer = gl.createBuffer();
    gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
    gl.enableVertexAttribArray( this.positionLocation );
    gl.vertexAttribPointer( this.positionLocation, 2, gl.FLOAT, false, 0, 0 );

    gl.blendFunc( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA );
    gl.enable( this.gl.BLEND );
  }

  return inherit( Object, TextureRenderer, {
    draw: function() {
      var gl = this.gl;

      gl.useProgram( this.colorShaderProgram );
      gl.enableVertexAttribArray( this.texCoordLocation );
      gl.enableVertexAttribArray( this.positionLocation );
      gl.enableVertexAttribArray( this.transform1AttributeLocation );
      gl.enableVertexAttribArray( this.transform2AttributeLocation );

      // Create a texture.
      gl.bindTexture( gl.TEXTURE_2D, this.texture );

      // set the resolution
      //TODO: This backing scale multiply seems very buggy and contradicts everything we know!
      // Still, it gives the right behavior on iPad3 and OSX (non-retina).  Should be discussed and investigated.
      gl.uniform2f( this.resolutionLocation, this.canvas.width / this.backingScale, this.canvas.height / this.backingScale );

      var step = Float32Array.BYTES_PER_ELEMENT;
      var total = 2 + 2 + 3 + 3;
      var stride = step * total;

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      gl.vertexAttribPointer( this.positionLocation, 2, gl.FLOAT, false, stride, 0 );
      gl.vertexAttribPointer( this.texCoordLocation, 2, gl.FLOAT, false, stride, step * 2 );
      gl.vertexAttribPointer( this.transform1AttributeLocation, 3, gl.FLOAT, false, stride, step * (2 + 2) );
      gl.vertexAttribPointer( this.transform2AttributeLocation, 3, gl.FLOAT, false, stride, step * (2 + 2 + 3) );

      // Draw the rectangle.
      gl.drawArrays( gl.TRIANGLES, 0, this.textureBufferData.vertexArray.length / total );

      gl.disableVertexAttribArray( this.texCoordLocation );
      gl.disableVertexAttribArray( this.positionLocation );
      gl.disableVertexAttribArray( this.transform1AttributeLocation );
      gl.disableVertexAttribArray( this.transform2AttributeLocation );

      gl.bindTexture( gl.TEXTURE_2D, null );
    },

    /**
     * Iterate through all of the sprite sheets and register the dirty ones with the GPU as texture units.
     */
    bindDirtyTextures: function() {
      var gl = this.gl;
      for ( var i = 0; i < this.textureBufferData.spriteSheets.length; i++ ) {
        var spriteSheet = this.textureBufferData.spriteSheets[i];
        if ( spriteSheet.dirty ) {
          gl.bindTexture( gl.TEXTURE_2D, this.texture );

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
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      // Keep track of the vertexArray for updating sublists of it
      this.vertexArray = new Float32Array( this.textureBufferData.vertexArray );
      gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW );
    },

    updateTriangleBuffer: function( geometry ) {
      var gl = this.gl;

      // Update the vertex locations
      // Use a buffer view to only update the changed vertices
      // like //see http://stackoverflow.com/questions/19892022/webgl-optimizing-a-vertex-buffer-that-changes-values-vertex-count-every-frame
      // See also http://stackoverflow.com/questions/5497722/how-can-i-animate-an-object-in-webgl-modify-specific-vertices-not-full-transfor
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      //Update the Float32Array values
      for ( var i = geometry.index; i < geometry.endIndex; i++ ) {
        this.vertexArray[i] = this.textureBufferData.vertexArray[i];
      }

      // Isolate the subarray of changed values
      var subArray = this.vertexArray.subarray( geometry.index, geometry.endIndex );

      // Send new values to the GPU
      // See https://www.khronos.org/webgl/public-mailing-list/archives/1201/msg00110.html
      // The the offset is the index times the bytes per value
      gl.bufferSubData( gl.ARRAY_BUFFER, geometry.index * 4, subArray );
    }
  } );
} );