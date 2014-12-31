//  Copyright 2002-2014, University of Colorado Boulder

/**
 * ColorTriangleRenderer manages the program & attributes & drawing for rendering indepdent triangles.  Geometry +data provided
 * by colorTriangleBufferData.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var ColorTriangleBufferData = require( 'SCENERY/display/webgl/colorTriangleBufferData' );

  // shaders
  var colorVertexShader = require( 'text!SCENERY/display/webgl/color2d.vert' );
  var colorFragmentShader = require( 'text!SCENERY/display/webgl/color2d.frag' );

  /**
   *
   * @constructor
   */
  function ColorTriangleRenderer( gl, backingScale, canvas ) {
    this.gl = gl;
    this.canvas = canvas;

    // Manages the indices within a single array, so that disjoint geometries can be represented easily here.
    // TODO: Compare this same idea to triangle strips
    this.colorTriangleBufferData = new ColorTriangleBufferData();

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
    gl.attachShader( this.colorShaderProgram, toShader( colorVertexShader, gl.VERTEX_SHADER, "VERTEX" ) );
    gl.attachShader( this.colorShaderProgram, toShader( colorFragmentShader, gl.FRAGMENT_SHADER, "FRAGMENT" ) );
    gl.linkProgram( this.colorShaderProgram );

    this.positionAttribLocation = gl.getAttribLocation( this.colorShaderProgram, 'aPosition' );
    this.colorAttributeLocation = gl.getAttribLocation( this.colorShaderProgram, 'aVertexColor' );

    gl.enableVertexAttribArray( this.positionAttribLocation );
    gl.enableVertexAttribArray( this.colorAttributeLocation );
    gl.useProgram( this.colorShaderProgram );

    // set the resolution
    var resolutionLocation = gl.getUniformLocation( this.colorShaderProgram, 'uResolution' );

    //TODO: This backing scale multiply seems very buggy and contradicts everything we know!
    // Still, it gives the right behavior on iPad3 and OSX (non-retina).  Should be discussed and investigated.
    gl.uniform2f( resolutionLocation, canvas.width / backingScale, canvas.height / backingScale );

    this.vertexBuffer = gl.createBuffer();
    this.bindVertexBuffer();

    this.depthBuffer = gl.createBuffer();
    this.bindDepthBuffer();

    // Set up different colors for each triangle
    this.vertexColorBuffer = gl.createBuffer();
    this.bindColorBuffer();

    gl.clearColor( 0.0, 0.0, 0.0, 0.0 );
    gl.enable( gl.DEPTH_TEST );
  }

  return inherit( Object, ColorTriangleRenderer, {

    draw: function() {
      var gl = this.gl;

      gl.enableVertexAttribArray( this.positionAttribLocation );
      gl.enableVertexAttribArray( this.colorAttributeLocation );
      gl.useProgram( this.colorShaderProgram );


      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      gl.vertexAttribPointer( this.positionAttribLocation, 2, gl.FLOAT, false, 0, 0 );

      // Send the colors to the GPU
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexColorBuffer );
      gl.vertexAttribPointer( this.colorAttributeLocation, 4, gl.FLOAT, false, 0, 0 );


      gl.drawArrays( gl.TRIANGLES, 0, this.colorTriangleBufferData.vertexArray.length / 2 );

      gl.disableVertexAttribArray( this.positionAttribLocation );
      gl.disableVertexAttribArray( this.colorAttributeLocation );
    },

    bindVertexBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      // Keep track of the vertexArray for updating sublists of it
      this.vertexArray = new Float32Array( this.colorTriangleBufferData.vertexArray );
      gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW );
    },

    bindColorBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexColorBuffer );
      gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( this.colorTriangleBufferData.colors ), gl.STATIC_DRAW );
    },

    bindDepthBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.depthBuffer );
      this.depthArray = new Float32Array( this.colorTriangleBufferData.depthBufferArray );
      gl.bufferData( gl.ARRAY_BUFFER, this.depthArray, gl.STATIC_DRAW );
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
        this.vertexArray[i] = this.colorTriangleBufferData.vertexArray[i];
      }

      // Isolate the subarray of changed values
      var subArray = this.vertexArray.subarray( geometry.index, geometry.endIndex );

      // Send new values to the GPU
      // See https://www.khronos.org/webgl/public-mailing-list/archives/1201/msg00110.html
      // The the offset is the index times the bytes per value
      gl.bufferSubData( gl.ARRAY_BUFFER, geometry.index * 4, subArray );
    }
  } );
} );