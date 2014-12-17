//  Copyright 2002-2014, University of Colorado Boulder

/**
 * Simplified isolated test harness for a webgl renderer.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var TriangleSystem = require( 'SCENERY/display/webgl/TriangleSystem' );
  var Events = require( 'AXON/Events' );

  // shaders
  var vertexShaderSource = require( 'text!SCENERY/display/webgl/color2d.vert' );
  var fragmentShaderSource = require( 'text!SCENERY/display/webgl/color2d.frag' );

  /**
   *
   * @constructor
   */
  function TestWebGL() {

    this.events = new Events();

    this.stats = this.createStats();
    document.body.appendChild( this.stats.domElement );

    var canvas = document.getElementById( "canvas" );

    // Handle retina displays as described in https://www.khronos.org/webgl/wiki/HandlingHighDPI
    // First, set the display size of the canvas.
    canvas.style.width = window.innerWidth + "px";
    canvas.style.height = window.innerHeight + "px";

    // Next, set the size of the drawingBuffer
    var devicePixelRatio = window.devicePixelRatio || 1;
    canvas.width = window.innerWidth * devicePixelRatio;
    canvas.height = window.innerHeight * devicePixelRatio;

    // Code inspired by http://www.webglacademy.com/#1
    var gl;
    try {
      gl = canvas.getContext( "experimental-webgl", {antialias: true} ); // TODO: {antialias:true?}
    }
    catch( e ) {
      return false;
    }
    this.gl = gl;

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

    var vertexShader = toShader( vertexShaderSource, gl.VERTEX_SHADER, "VERTEX" );
    var fragmentShader = toShader( fragmentShaderSource, gl.FRAGMENT_SHADER, "FRAGMENT" );

    var shaderProgram = gl.createProgram();
    gl.attachShader( shaderProgram, vertexShader );
    gl.attachShader( shaderProgram, fragmentShader );

    gl.linkProgram( shaderProgram );

    this.positionAttribLocation = gl.getAttribLocation( shaderProgram, 'aPosition' );
    this.colorAttributeLocation = gl.getAttribLocation( shaderProgram, 'aVertexColor' );

    gl.enableVertexAttribArray( this.positionAttribLocation );
    gl.enableVertexAttribArray( this.colorAttributeLocation );

    gl.useProgram( shaderProgram );

    // Manages the indices within a single array, so that disjoint geometries can be represented easily here.
    // TODO: Compare this same idea to triangle strips
    this.trianglesGeometry = new TriangleSystem();

    this.vertexBuffer = gl.createBuffer();
    this.bindVertexBuffer();

    // Set up different colors for each triangle
    this.vertexColorBuffer = gl.createBuffer();
    this.bindColorBuffer();

    gl.clearColor( 0.0, 0.0, 0.0, 0.0 );

    this.boundAnimate = this.animate.bind( this );
  }

  return inherit( Object, TestWebGL, {

    /**
     * Create a mrdoob stats instance which can be used to profile the simulation.
     * @returns {Stats}
     */
    createStats: function() {
      var stats = new Stats();
      stats.setMode( 0 ); // 0: fps, 1: ms

      // align top-left
      stats.domElement.style.position = 'absolute';
      stats.domElement.style.left = '0px';
      stats.domElement.style.top = '0px';

      return stats;
    },

    /**
     * Initialize the simulation and start it animating.
     */
    start: function() {
      window.requestAnimationFrame( this.boundAnimate );
    },

    animate: function() {
      this.stats.begin();
      window.requestAnimationFrame( this.boundAnimate );

      this.events.trigger( 'step' );

      var gl = this.gl;

      gl.viewport( 0.0, 0.0, canvas.width, canvas.height );
      gl.clear( gl.COLOR_BUFFER_BIT );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      // Update the vertex locations
      //see http://stackoverflow.com/questions/5497722/how-can-i-animate-an-object-in-webgl-modify-specific-vertices-not-full-transfor
      //TODO: Use a buffer view to only update the changed vertices
      gl.bufferSubData( gl.ARRAY_BUFFER, 0, new Float32Array( this.trianglesGeometry.vertexArray ) );
      gl.vertexAttribPointer( this.positionAttribLocation, 2, gl.FLOAT, false, 0, 0 );

      // Send the colors to the GPU
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexColorBuffer );
      gl.vertexAttribPointer( this.colorAttributeLocation, 4, gl.FLOAT, false, 0, 0 );

      gl.drawArrays( gl.TRIANGLES, 0, this.trianglesGeometry.vertexArray.length / 2 );
      gl.flush();

      this.stats.end();
    },

    bindVertexBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( this.trianglesGeometry.vertexArray ), gl.DYNAMIC_DRAW );
    },

    bindColorBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexColorBuffer );
      gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( this.trianglesGeometry.colors ), gl.STATIC_DRAW );
    }

  } );
} );