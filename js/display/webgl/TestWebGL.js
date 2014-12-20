//  Copyright 2002-2014, University of Colorado Boulder

/**
 * Simplified isolated test harness for a webgl renderer.
 *
 * TODO: Array of structures for interleaved vertex data (color + texture coordinates)
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var TriangleSystem = require( 'SCENERY/display/webgl/TriangleSystem' );
  var Events = require( 'AXON/Events' );
  var Util = require( 'SCENERY/util/Util' );

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


    this.canvas = document.createElement( "canvas" );
    this.canvas.style.position = 'absolute';
    this.canvas.style.left = '0';
    this.canvas.style.top = '0';
    this.canvas.style.pointerEvents = 'none';

    document.body.appendChild( this.canvas );
    document.body.appendChild( this.stats.domElement );

    // Code inspired by http://www.webglacademy.com/#1
    var gl;
    try {
      gl = this.canvas.getContext( "experimental-webgl", {antialias: true} ); // TODO: {antialias:true?}
    }
    catch( e ) {
      return false;
    }
    this.gl = gl;

    // Handle retina displays as described in https://www.khronos.org/webgl/wiki/HandlingHighDPI
    // First, set the display size of the canvas.
    this.canvas.style.width = window.innerWidth + "px";
    this.canvas.style.height = window.innerHeight + "px";

    // Next, set the size of the drawingBuffer
    var backingScale = Util.backingScale( this.gl );
    var devicePixelRatio = window.devicePixelRatio || 1;
    this.canvas.width = window.innerWidth * devicePixelRatio;
    this.canvas.height = window.innerHeight * devicePixelRatio;

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

    // set the resolution
    var resolutionLocation = gl.getUniformLocation( shaderProgram, 'uResolution' );

    //TODO: This backing scale multiply seems very buggy and contradicts everything we know!
    // Still, it gives the right behavior on iPad3 and OSX (non-retina).  Should be discussed and investigated.
    gl.uniform2f( resolutionLocation, this.canvas.width / backingScale, this.canvas.height / backingScale );

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

      // Keep track of the time for profiling
      this.stats.begin();

      // Queue the next animation frame
      window.requestAnimationFrame( this.boundAnimate );

      // Let listeners update their state
      this.events.trigger( 'step' );

      // Render everything
      this.draw();

      // Record the timing for @mrdoob stats profiler
      this.stats.end();
    },
    draw: function() {
      var gl = this.gl;

      gl.viewport( 0.0, 0.0, this.canvas.width, this.canvas.height );
      gl.clear( gl.COLOR_BUFFER_BIT );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      gl.vertexAttribPointer( this.positionAttribLocation, 2, gl.FLOAT, false, 0, 0 );

      // Send the colors to the GPU
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexColorBuffer );
      gl.vertexAttribPointer( this.colorAttributeLocation, 4, gl.FLOAT, false, 0, 0 );

      gl.drawArrays( gl.TRIANGLES, 0, this.trianglesGeometry.vertexArray.length / 2 );
      gl.flush();
    },

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
//      gl.bufferSubData( gl.ARRAY_BUFFER, 0, new Float32Array( this.trianglesGeometry.vertexArray ) );
//    },

    updateTriangleBuffer: function( geometry ) {
      var gl = this.gl;

      // Update the vertex locations
      // Use a buffer view to only update the changed vertices
      // like //see http://stackoverflow.com/questions/19892022/webgl-optimizing-a-vertex-buffer-that-changes-values-vertex-count-every-frame
      // See also http://stackoverflow.com/questions/5497722/how-can-i-animate-an-object-in-webgl-modify-specific-vertices-not-full-transfor
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      //Update the Float32Array values
      for ( var i = geometry.index; i < geometry.endIndex; i++ ) {
        this.vertexArray[i] = this.trianglesGeometry.vertexArray[i];
      }

      // Isolate the subarray of changed values
      var subArray = this.vertexArray.subarray( geometry.index, geometry.endIndex );

      // Send new values to the GPU
      // See https://www.khronos.org/webgl/public-mailing-list/archives/1201/msg00110.html
      // The the offset is the index times the bytes per value
      gl.bufferSubData( gl.ARRAY_BUFFER, geometry.index * 4, subArray );

//      console.log(
//        'vertex array length', this.trianglesGeometry.vertexArray.length,
//        'va.length', this.vertexArray.length,
//        'geometry index', geometry.index,
//        'geometry end index', geometry.endIndex,
//        'updated size', subArray.length );

    },

    bindVertexBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      // Keep track of the vertexArray for updating sublists of it
      this.vertexArray = new Float32Array( this.trianglesGeometry.vertexArray );
      gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW );
    },

    bindColorBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexColorBuffer );
      gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( this.trianglesGeometry.colors ), gl.STATIC_DRAW );
    }

  } );
} );