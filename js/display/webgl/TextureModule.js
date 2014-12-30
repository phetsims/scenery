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
  var TriangleSystem = require( 'SCENERY/display/webgl/TriangleSystem' );

  var colorVertexShader = require( 'text!SCENERY/display/webgl/texture.vert' );
  var colorFragmentShader = require( 'text!SCENERY/display/webgl/texture.frag' );

  function setRectangle( gl, x, y, width, height ) {
    var x1 = x;
    var x2 = x + width;
    var y1 = y;
    var y2 = y + height;
    gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( [
      x1, y1,
      x2, y1,
      x1, y2,
      x1, y2,
      x2, y1,
      x2, y2] ), gl.STATIC_DRAW );
  }

  /**
   *
   * @constructor
   */
  function TextureModule( gl, backingScale, canvas ) {
    var textureModule = this;
    this.gl = gl;
    this.canvas = canvas;

    // Manages the indices within a single array, so that disjoint geometries can be represented easily here.
    // TODO: Compare this same idea to triangle strips
    this.triangleSystem = new TriangleSystem();

    var toShader = function( source, type, typeString ) {
      var shader = gl.createShader( type );
      gl.shaderSource( shader, source );
      gl.compileShader( shader );
      if ( !gl.getShaderParameter( shader, gl.COMPILE_STATUS ) ) {
        console.log( "ERROR IN " + typeString + " SHADER : " + gl.getShaderInfoLog( shader ) );
        return false;
      }
      return shader;
    };

    this.colorShaderProgram = gl.createProgram();
    var program = this.colorShaderProgram;
    gl.attachShader( this.colorShaderProgram, toShader( colorVertexShader, gl.VERTEX_SHADER, "VERTEX" ) );
    gl.attachShader( this.colorShaderProgram, toShader( colorFragmentShader, gl.FRAGMENT_SHADER, "FRAGMENT" ) );
    gl.linkProgram( this.colorShaderProgram );

    // look up where the vertex data needs to go.
    this.positionLocation = gl.getAttribLocation( program, "a_position" );
    this.texCoordLocation = gl.getAttribLocation( program, "a_texCoord" );

    // provide texture coordinates for the rectangle.
    this.texCoordBuffer = gl.createBuffer();
    gl.bindBuffer( gl.ARRAY_BUFFER, this.texCoordBuffer );
    gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( [
      0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0,
      0.0, 1.0,
      1.0, 0.0,
      1.0, 1.0] ), gl.STATIC_DRAW );
    gl.enableVertexAttribArray( this.texCoordLocation );
    gl.vertexAttribPointer( this.texCoordLocation, 2, gl.FLOAT, false, 0, 0 );

    // Create a texture.
    this.texture = gl.createTexture();

    // lookup uniforms
    this.resolutionLocation = gl.getUniformLocation( program, "u_resolution" );

    // set the resolution
//    gl.uniform2f( this.resolutionLocation, this.canvas.width, this.canvas.height );

    // Create a buffer for the position of the rectangle corners.
    this.buffer = gl.createBuffer();
    gl.bindBuffer( gl.ARRAY_BUFFER, this.buffer );
    gl.enableVertexAttribArray( this.positionLocation );
    gl.vertexAttribPointer( this.positionLocation, 2, gl.FLOAT, false, 0, 0 );

    setRectangle( gl, 0, 0, 256, 256 );

    // TODO: only create once instance of this Canvas for reuse
    this.image = document.createElement( 'canvas' );
    this.image.width = 256;
    this.image.height = 256;
    var context = this.image.getContext( '2d' );

    var loadedImage = new Image();
    loadedImage.src = "http://localhost:8080/energy-skate-park-basics/images/mountains.png";  // MUST BE SAME DOMAIN!!!
    loadedImage.onload = function() {
      context.drawImage( loadedImage, 0, 0 );

      // Set a rectangle the same size as the image.
      setRectangle( gl, 0, 0, textureModule.image.width, textureModule.image.height );
    };
  }

  return inherit( Object, TextureModule, {
    draw: function() {
      var gl = this.gl;

      gl.useProgram( this.colorShaderProgram );
      gl.enableVertexAttribArray( this.texCoordLocation );
      gl.enableVertexAttribArray( this.positionLocation );

      // provide texture coordinates for the rectangle.
      gl.bindBuffer( gl.ARRAY_BUFFER, this.texCoordBuffer );
      gl.vertexAttribPointer( this.texCoordLocation, 2, gl.FLOAT, false, 0, 0 );

      // Create a texture.
      gl.bindTexture( gl.TEXTURE_2D, this.texture );

      // Set the parameters so we can render any size image.
      gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
      gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );
      gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST );
      gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST );

      // Upload the image into the texture.
      gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.image );

      // set the resolution
      gl.uniform2f( this.resolutionLocation, this.canvas.width, this.canvas.height );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.buffer );

      gl.vertexAttribPointer( this.positionLocation, 2, gl.FLOAT, false, 0, 0 );

      // Draw the rectangle.
      gl.drawArrays( gl.TRIANGLES, 0, 6 );

      gl.disableVertexAttribArray( this.texCoordLocation );
      gl.disableVertexAttribArray( this.positionLocation );

    },
    bindVertexBuffer: function() {
//      var gl = this.gl;
//      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
//
//      // Keep track of the vertexArray for updating sublists of it
//      this.vertexArray = new Float32Array( this.triangleSystem.vertexArray );
//      gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW );
    },

    bindColorBuffer: function() {
//      var gl = this.gl;
//      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexColorBuffer );
//      gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( this.triangleSystem.colors ), gl.STATIC_DRAW );
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
        this.vertexArray[i] = this.triangleSystem.vertexArray[i];
      }

      // Isolate the subarray of changed values
      var subArray = this.vertexArray.subarray( geometry.index, geometry.endIndex );

      // Send new values to the GPU
      // See https://www.khronos.org/webgl/public-mailing-list/archives/1201/msg00110.html
      // The the offset is the index times the bytes per value
      gl.bufferSubData( gl.ARRAY_BUFFER, geometry.index * 4, subArray );

//      console.log(
//        'vertex array length', this.triangleSystem.vertexArray.length,
//        'va.length', this.vertexArray.length,
//        'geometry index', geometry.index,
//        'geometry end index', geometry.endIndex,
//        'updated size', subArray.length );

    }


    /**
     * Update all of the vertices in the entire triangles geometry.  Probably just faster
     * to update the changed vertices.  Use this if many things changed, though.
     * @private
     */
//    bufferSubData: function() {
//      var gl = this.gl;
//
//      // Update the vertex locations
//      //see http://stackoverflow.com/questions/5497722/how-can-i-animate-an-object-in-webgl-modify-specific-vertices-not-full-transfor
//      //TODO: Use a buffer view to only update the changed vertices
//      //perhaps like //see http://stackoverflow.com/questions/19892022/webgl-optimizing-a-vertex-buffer-that-changes-values-vertex-count-every-frame
//      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
//      gl.bufferSubData( gl.ARRAY_BUFFER, 0, new Float32Array( this.triangleSystem.vertexArray ) );
//    },

  } );
} );