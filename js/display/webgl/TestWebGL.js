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
  var vertexShaderSource = require( 'text!SCENERY/display/webgl/color2d.vert' );
  var fragmentShaderSource = require( 'text!SCENERY/display/webgl/color2d.frag' );

  /**
   *
   * @constructor
   */
  function TestWebGL() {

    var testWebGL = this;

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
    var vertexArray = this.trianglesGeometry.vertexArray;
    var colors = this.trianglesGeometry.colors;

    this.rectangles = [];

    var numRectangles = 500;
    for ( var i = 0; i < numRectangles; i++ ) {
      this.addRectangle();
    }

    var numStars = 500;
    this.stars = [];
    for ( var k = 0; k < numStars; k++ ) {
      this.addStar();
    }

    document.getElementById( 'add-rectangle' ).onclick = function() {
      testWebGL.addRectangle();
    };

    this.vertexBuffer = gl.createBuffer();
    gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
    gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( vertexArray ), gl.DYNAMIC_DRAW );

    // Set up different colors for each triangle
    this.vertexColorBuffer = gl.createBuffer();
    gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexColorBuffer );
    gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( colors ), gl.STATIC_DRAW );

    gl.clearColor( 0.0, 0.0, 0.0, 0.0 );

    this.gl = gl;
    this.boundAnimate = this.animate.bind( this );

    this.vertexArray = vertexArray;
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
      window.requestAnimationFrame( this.boundAnimate );
      var gl = this.gl;

      this.stats.begin();

      gl.viewport( 0.0, 0.0, canvas.width, canvas.height );
      gl.clear( gl.COLOR_BUFFER_BIT );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      // Update the vertex locations
      //see http://stackoverflow.com/questions/5497722/how-can-i-animate-an-object-in-webgl-modify-specific-vertices-not-full-transfor
      gl.bufferSubData( gl.ARRAY_BUFFER, 0, new Float32Array( this.vertexArray ) );
      gl.vertexAttribPointer( this.positionAttribLocation, 2, gl.FLOAT, false, 0, 0 );

      // Send the colors to the GPU
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexColorBuffer );
      gl.vertexAttribPointer( this.colorAttributeLocation, 4, gl.FLOAT, false, 0, 0 );

      // Show one oscillation per second so it is easy to count time
      var x = 0.2 * Math.cos( Date.now() / 1000 * 2 * Math.PI );
      for ( var i = 0; i < this.rectangles.length; i++ ) {
        var rectangle = this.rectangles[i];
        rectangle.setXWidth( rectangle.initialState.x + x, rectangle.initialState.width );
      }

      for ( var mm = 0; mm < this.stars.length / 2; mm++ ) {
        var star = this.stars[mm];
        star.setStar( star.initialState._x, star.initialState._y, star.initialState._innerRadius, star.initialState._outerRadius, star.initialState._totalAngle + Date.now() / 1000 );
      }

      gl.drawArrays( gl.TRIANGLES, 0, this.vertexArray.length / 2 );
      gl.flush();

      this.stats.end();
    },

    addRectangle: function() {
      var x = (Math.random() * 2 - 1) * 0.9;
      var y = (Math.random() * 2 - 1) * 0.9;
      this.rectangles.push( this.trianglesGeometry.createRectangle( x, y, 0.02, 0.02, x, y, 1, 1 ) );
    },

    addStar: function() {
      var x = (Math.random() * 2 - 1) * 0.9;
      var y = (Math.random() * 2 - 1) * 0.9;
      var scale = Math.random() * 0.2;
      var star = this.trianglesGeometry.createStar( x, y, 0.15 * scale, 0.4 * scale, Math.PI + Math.random() * Math.PI * 2, Math.random(), Math.random(), Math.random(), 1 );
      this.stars.push( star );
    }
  } );
} );