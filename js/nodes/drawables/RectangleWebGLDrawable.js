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

      this.buffer = gl.createBuffer();
      gl.bindBuffer( gl.ARRAY_BUFFER, this.buffer );
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array( [
          -1.0, -1.0,
          1.0, -1.0,
          -1.0, 1.0,
          -1.0, 1.0,
          1.0, -1.0,
          1.0, 1.0] ),
        gl.STATIC_DRAW );

      this.updateRectangle();
    },

    updateRectangle: function() {
    },

    render: function( shaderProgram, viewMatrix ) {
      var gl = this.gl;

      //TODO: Transform the rectangle
//      var uMatrix = viewMatrix.timesMatrix( Matrix4.scaling( this.canvasWidth, -this.canvasHeight, 1 ).timesMatrix( Matrix4.translation( 0, -1 ) ) );

      // look up where the vertex data needs to go.
      var positionLocation = gl.getAttribLocation( shaderProgram.program, "a_position" );

      // Create a buffer and put a single clipspace rectangle in it (2 triangles)
      gl.enableVertexAttribArray( positionLocation );
      gl.vertexAttribPointer( positionLocation, 2, gl.FLOAT, false, 0, 0 );

      // draw
      gl.drawArrays( gl.TRIANGLES, 0, 6 );
    },

    dispose: function() {
      this.gl.deleteBuffer( this.buffer );
    }
  } );
} );