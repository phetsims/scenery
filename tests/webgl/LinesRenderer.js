// Copyright 2002-2014, University of Colorado Boulder

/**
 * Demonstration of a custom WebGL renderer (like CanvasNode), in this case which draws lines.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var WebGLUtil = require( 'SCENERY/display/webgl/WebGLUtil' );

  // shaders
  var lineVertexShader = require( 'text!SCENERY/../tests/webgl/lines.vert' );
  var lineFragmentShader = require( 'text!SCENERY/../tests/webgl/lines.frag' );

  /**
   *
   * @constructor
   */
  function LinesRenderer( gl, backingScale, canvas ) {
  }

  return inherit( Object, LinesRenderer, {
    //webGLRenderer.gl, webGLRenderer.backingScale, webGLRenderer.canvas 
    init: function( gl, backingScale, canvas ) {
      this.gl = gl;
      this.canvas = canvas;

      // Manages the indices within a single array, so that disjoint geometries can be represented easily here.
      // TODO: Compare this same idea to triangle strips

      this.lineShaderProgram = gl.createProgram();
      gl.attachShader( this.lineShaderProgram, WebGLUtil.toShader( gl, lineVertexShader, gl.VERTEX_SHADER, 'VERTEX' ) );
      gl.attachShader( this.lineShaderProgram, WebGLUtil.toShader( gl, lineFragmentShader, gl.FRAGMENT_SHADER, 'FRAGMENT' ) );
      gl.linkProgram( this.lineShaderProgram );

      this.positionAttribLocation = gl.getAttribLocation( this.lineShaderProgram, 'aPosition' );

      gl.enableVertexAttribArray( this.positionAttribLocation );
      gl.useProgram( this.lineShaderProgram );

      this.vertexBuffer = gl.createBuffer();
      this.bindVertexBuffer();
    },
    draw: function() {
      var gl = this.gl;

      gl.enableVertexAttribArray( this.positionAttribLocation );
      gl.useProgram( this.lineShaderProgram );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      gl.vertexAttribPointer( this.positionAttribLocation, 3, gl.FLOAT, false, 0, 0 );

      gl.drawArrays( gl.LINES, 0, 2 );

      gl.disableVertexAttribArray( this.positionAttribLocation );
    },

    bindVertexBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( [
        0, 0, 0.1,
        0.5, 0.5, 0.1
      ] ), gl.STATIC_DRAW );
    }
  } );
} );