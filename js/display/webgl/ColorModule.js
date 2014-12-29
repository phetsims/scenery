//  Copyright 2002-2014, University of Colorado Boulder

/**
 * ColorModule manages the program & attributes & drawing for rendering indepdent triangles.  Geometry +data provided
 * by TriangleSystem.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var TriangleSystem = require( 'SCENERY/display/webgl/TriangleSystem' );

  var colorVertexShader = require( 'text!SCENERY/display/webgl/color2d.vert' );
  var colorFragmentShader = require( 'text!SCENERY/display/webgl/color2d.frag' );

  /**
   *
   * @constructor
   */
  function ColorModule( gl, backingScale, canvas ) {
    this.gl = gl;
    this.canvas = canvas;

    // Manages the indices within a single array, so that disjoint geometries can be represented easily here.
    // TODO: Compare this same idea to triangle strips
    this.triangleSystem = new TriangleSystem();

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

    // Set up different colors for each triangle
    this.vertexColorBuffer = gl.createBuffer();
    this.bindColorBuffer();

    gl.clearColor( 0.0, 0.0, 0.0, 0.0 );
  }

  return inherit( Object, ColorModule, {
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

      gl.drawArrays( gl.TRIANGLES, 0, this.triangleSystem.vertexArray.length / 2 );

      gl.disableVertexAttribArray( this.positionAttribLocation );
      gl.disableVertexAttribArray( this.colorAttributeLocation );
    },
    bindVertexBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      // Keep track of the vertexArray for updating sublists of it
      this.vertexArray = new Float32Array( this.triangleSystem.vertexArray );
      gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW );
    },

    bindColorBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexColorBuffer );
      gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( this.triangleSystem.colors ), gl.STATIC_DRAW );
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
        this.vertexArray[i] = this.triangleSystem.vertexArray[i];
      }

      // Isolate the subarray of changed values
      var subArray = this.vertexArray.subarray( geometry.index, geometry.endIndex );

      // Send new values to the GPU
      // See https://www.khronos.org/webgl/public-mailing-list/archives/1201/msg00110.html
      // The the offset is the index times the bytes per value
      gl.bufferSubData( gl.ARRAY_BUFFER, geometry.index * 4, subArray );

//      console.log(
//        'vertex array length', this.triangleSystem.vertexArray.length,
//        'va.length', this.vertexArray.length,
//        'geometry index', geometry.index,
//        'geometry end index', geometry.endIndex,
//        'updated size', subArray.length );

    }


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
//      gl.bufferSubData( gl.ARRAY_BUFFER, 0, new Float32Array( this.triangleSystem.vertexArray ) );
//    },

  } );
} );