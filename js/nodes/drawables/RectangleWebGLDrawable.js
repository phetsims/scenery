// Copyright 2002-2013, University of Colorado

/**
 * WebGL state for rendering Rectangle nodes
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix4 = require( 'DOT/Matrix4' );
  var scenery = require( 'SCENERY/scenery' );
  var Util = require( 'SCENERY/util/Util' );

  scenery.RectangleWebGLDrawable = function RectangleWebGLDrawable( gl, rectangleNode ) {
    this.rectangleNode = rectangleNode;

    this.gl = gl;
    this.initialize( gl );
  };

  return inherit( Object, scenery.RectangleWebGLDrawable, {
    initialize: function() {
      var gl = this.gl;

      this.texture = null;

      var vertexBuffer = this.vertexBuffer = gl.createBuffer();
      gl.bindBuffer( gl.ARRAY_BUFFER, vertexBuffer );
      gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( [
        0, 0,
        0, 1,
        1, 0,
        1, 1
      ] ), gl.STATIC_DRAW );

      this.updateRectangle();
    },

    updateRectangle: function() {
      var gl = this.gl;

      if ( this.texture !== null ) {
        gl.deleteTexture( this.texture );
      }

      //TODO: Do not render rectangles as images
      var canvas = document.createElement( 'canvas' );
      var context = canvas.getContext( '2d' );
      this.canvasWidth = canvas.width = Util.toPowerOf2( this.rectangleNode.getWidth() );
      this.canvasHeight = canvas.height = Util.toPowerOf2( this.rectangleNode.getHeight() );
      context.fillStyle = "orange"; //Happy Halloween
      context.fillRect( 0, 0, this.rectangleNode.getWidth(), this.rectangleNode.getHeight() );

      var texture = this.texture = gl.createTexture();
      gl.bindTexture( gl.TEXTURE_2D, texture );
      gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE );
      gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE );
      gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR );
      gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR ); // TODO: better filtering
      gl.pixelStorei( gl.UNPACK_FLIP_Y_WEBGL, true );
      gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas );
      gl.bindTexture( gl.TEXTURE_2D, null );
    },

    render: function( shaderProgram, viewMatrix ) {
      var gl = this.gl;

      var uMatrix = viewMatrix.timesMatrix( Matrix4.scaling( this.canvasWidth, -this.canvasHeight, 1 ).timesMatrix( Matrix4.translation( 0, -1 ) ) );

      // combine image matrix (to scale aspect ratios), the trail's matrix, and the matrix to device coordinates
      gl.uniformMatrix4fv( shaderProgram.uniformLocations.uMatrix, false, uMatrix.entries );
      gl.uniform1i( shaderProgram.uniformLocations.uTexture, 0 ); // TEXTURE0 slot

      gl.activeTexture( gl.TEXTURE0 );
      gl.bindTexture( gl.TEXTURE_2D, this.texture );
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      gl.vertexAttribPointer( shaderProgram.attributeLocations.aVertex, 2, gl.FLOAT, false, 0, 0 );
      gl.drawArrays( gl.TRIANGLE_STRIP, 0, 4 );
      gl.bindTexture( gl.TEXTURE_2D, null );
    },

    dispose: function() {
      this.gl.deleteTexture( this.texture );
      this.gl.deleteBuffer( this.vertexBuffer );
    }
  } );
} );
