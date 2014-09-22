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
  var WebGLLayer = require( 'SCENERY/layers/WebGLLayer' );
  var Color = require( 'SCENERY/util/Color' );

  scenery.RectangleWebGLDrawable = function RectangleWebGLDrawable( gl, rectangleNode ) {
    this.rectangleNode = rectangleNode;

    this.gl = gl;
    this.initialize( gl );
  };

  return inherit( Object, scenery.RectangleWebGLDrawable, {
    initialize: function() {
      var gl = this.gl;

      this.buffer = gl.createBuffer();
      gl.bindBuffer( gl.ARRAY_BUFFER, this.buffer );
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array( [
          0, 0,
          1, 0,
          0, 1,
          1, 1] ),
        gl.STATIC_DRAW );

      this.updateRectangle();
    },

    updateRectangle: function() {
    },

    render: function( shaderProgram, viewMatrix ) {
      var gl = this.gl;

      var rectangleOffset = Matrix4.translation( this.rectangleNode.getRectX(), this.rectangleNode.getRectY(), 0 );
      var rectangleSize = Matrix4.scaling( this.rectangleNode.getWidth(), this.rectangleNode.getHeight(), 1 );
      var uMatrix = viewMatrix.timesMatrix( rectangleOffset.timesMatrix( rectangleSize ) );

      // combine image matrix (to scale aspect ratios), the trail's matrix, and the matrix to device coordinates
      gl.uniformMatrix4fv( shaderProgram.uniformLocations.uMatrix, false, uMatrix.entries );

      //Indicate the branch of logic to use in the ubershader.  In this case, a texture should be used for the image
      gl.uniform1i( shaderProgram.uniformLocations.uFragmentType, WebGLLayer.fragmentTypeFill );
      var color = Color.toColor( this.rectangleNode._fill );
      gl.uniform4f( shaderProgram.uniformLocations.uColor, color.r / 255, color.g / 255, color.b / 255, color.a );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.buffer );
      gl.vertexAttribPointer( shaderProgram.attributeLocations.aVertex, 2, gl.FLOAT, false, 0, 0 );
      gl.drawArrays( gl.TRIANGLE_STRIP, 0, 4 );
    },

    dispose: function() {
      this.gl.deleteBuffer( this.buffer );
    }
  } );
} );