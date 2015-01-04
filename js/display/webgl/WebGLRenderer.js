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
  var Events = require( 'AXON/Events' );
  var Util = require( 'SCENERY/util/Util' );
  var ColorTriangleRenderer = require( 'SCENERY/display/webgl/ColorTriangleRenderer' );
  var TextureRenderer = require( 'SCENERY/display/webgl/TextureRenderer' );

  /**
   *
   * @constructor
   */
  function WebGLRenderer() {

    this.events = new Events();

    this.stats = this.createStats();


    this.canvas = document.createElement( 'canvas' );
    this.canvas.style.position = 'absolute';
    this.canvas.style.left = '0';
    this.canvas.style.top = '0';
    this.canvas.style.pointerEvents = 'none';

    document.body.appendChild( this.canvas );
    document.body.appendChild( this.stats.domElement );

    // Code inspired by http://www.webglacademy.com/#1
    var gl;
    try {
      gl = this.canvas.getContext( 'experimental-webgl', {antialias: true} ); // TODO: {antialias:true?}
    }
    catch( e ) {
      return false;
    }
    this.gl = gl;

    // Handle retina displays as described in https://www.khronos.org/webgl/wiki/HandlingHighDPI
    // First, set the display size of the canvas.
    this.canvas.style.width = window.innerWidth + 'px';
    this.canvas.style.height = window.innerHeight + 'px';

    // Next, set the size of the drawingBuffer
    var backingScale = Util.backingScale( this.gl );
    var devicePixelRatio = window.devicePixelRatio || 1;
    this.canvas.width = window.innerWidth * devicePixelRatio;
    this.canvas.height = window.innerHeight * devicePixelRatio;

    this.colorTriangleRenderer = new ColorTriangleRenderer( gl, backingScale, this.canvas );
    this.textureRenderer = new TextureRenderer( gl, backingScale, this.canvas );
    this.customWebGLRenderers = [];

    this.boundAnimate = this.animate.bind( this );
    this.backingScale = backingScale;
  }

  return inherit( Object, WebGLRenderer, {
    addCustomWebGLRenderer: function( customWebGLRenderer ) {
      this.customWebGLRenderers.push( customWebGLRenderer );
    },
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
      gl.clear( gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT );

      //Render program by program.
      this.colorTriangleRenderer.draw();
      this.textureRenderer.draw();
      for ( var i = 0; i < this.customWebGLRenderers.length; i++ ) {
        this.customWebGLRenderers[i].draw();
      }

      //Flush after rendering complete.
      gl.flush();
    }
  } );
} );