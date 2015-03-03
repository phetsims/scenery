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
  function WebGLRenderer( options ) {
    // defaults
    this.options = _.extend( {
      stats: true,
      preserveDrawingBuffer: false
    }, options );

    this.events = new Events();

    // Create the stats and show it, but only for the standalone test cases (not during scenery usage).
    // TODO: A better design for stats vs no stats
    if ( this.options.stats ) {
      this.stats = this.createStats();
    }

    this.canvas = document.createElement( 'canvas' );
    this.canvas.style.position = 'absolute';
    this.canvas.style.left = '0';
    this.canvas.style.top = '0';
    this.canvas.style.pointerEvents = 'none';

    document.body.appendChild( this.canvas );
    if ( this.options.stats ) {
      document.body.appendChild( this.stats.domElement );
    }

    // Code inspired by http://www.webglacademy.com/#1
    var gl;
    try {
      // NOTE: If we add preserveDrawingBuffer, please add a clear() every frame
      gl = this.canvas.getContext( 'experimental-webgl', {
        antialias: true,
        preserveDrawingBuffer: this.options.preserveDrawingBuffer // major performance hit if true
      } );
    }
    catch( e ) {
      return false;
    }
    this.gl = gl;

    this.backingScale = Util.backingScale( this.gl );

  // TODO: When used by scenery, use different initial size (hopefully provided in the constructor args as an option)
    this.setCanvasSize( window.innerWidth, window.innerHeight );

    gl.clearColor( 0, 0, 0, 0 );
    gl.clear( gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT );
    gl.enable( gl.DEPTH_TEST );

    gl.blendFunc( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA );
    gl.enable( gl.BLEND );

    this.colorTriangleRenderer = new ColorTriangleRenderer( gl, this.backingScale, this.canvas );
    this.textureRenderer = new TextureRenderer( gl, this.backingScale, this.canvas );
    this.customWebGLRenderers = [];

    this.boundAnimate = this.animate.bind( this );
  }

  return inherit( Object, WebGLRenderer, {
    addCustomWebGLRenderer: function( customWebGLRenderer ) {
      customWebGLRenderer.init( this.gl, this.backingScale, this.canvas );
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
      if ( this.stats ) {
        this.stats.begin();
      }

      // Queue the next animation frame
      window.requestAnimationFrame( this.boundAnimate );

      // Let listeners update their state
      this.events.trigger( 'step' );

      // Render everything
      this.draw();

      // Record the timing for @mrdoob stats profiler
      if ( this.stats ) {
        this.stats.end();
      }
    },
    draw: function() {
      var gl = this.gl;

      // clearing the buffer is only needed if we gave the flag to preserve it
      if ( this.options.preserveDrawingBuffer ) {
        gl.clear( gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT );
      }

      gl.viewport( 0.0, 0.0, this.canvas.width, this.canvas.height );

      //Render program by program.
      this.colorTriangleRenderer.draw();
      this.textureRenderer.draw();
      for ( var i = 0; i < this.customWebGLRenderers.length; i++ ) {
        this.customWebGLRenderers[ i ].draw();
      }

      //Flush after rendering complete.
      gl.flush();
    },
    setCanvasSize: function( width, height ) {
      // Handle retina displays as described in https://www.khronos.org/webgl/wiki/HandlingHighDPI
      // First, set the display size of the canvas.
      this.canvas.style.width = width + 'px';
      this.canvas.style.height = height + 'px';

      // Next, set the size of the drawingBuffer
      var devicePixelRatio = window.devicePixelRatio || 1;
      this.canvas.width = width * devicePixelRatio;
      this.canvas.height = height * devicePixelRatio;
    },
    dispose: function() {
      //TODO: Dispose of more things!
      this.canvas.width = 0;
      this.canvas.height = 0;
    }
  } );
} );