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
  var ShaderProgram = require( 'SCENERY/util/ShaderProgram' );

  // shaders
  var colorVertexShaderSource = require( 'text!SCENERY/display/webgl/colorTriangle.vert' );
  var colorFragmentShaderSource = require( 'text!SCENERY/display/webgl/colorTriangle.frag' );

  /**
   *
   * @constructor
   */
  function ColorTriangleRenderer( gl, backingScale, canvas ) {
    this.gl = gl;
    this.canvas = canvas;
    this.backingScale = backingScale;

    // Manages the indices within a single array, so that disjoint geometries can be represented easily here.
    // TODO: Compare this same idea to triangle strips
    this.colorTriangleBufferData = new ColorTriangleBufferData();

    this.shaderProgram = new ShaderProgram( gl, colorVertexShaderSource, colorFragmentShaderSource, {
      attributes: [ 'aPosition', 'aVertexColor', 'aTransform1', 'aTransform2' ],
      uniforms: [ 'uResolution' ]
    } );

    this.vertexBuffer = gl.createBuffer();
    this.bindVertexBuffer();
  }

  return inherit( Object, ColorTriangleRenderer, {

    draw: function() {

      if ( this.colorTriangleBufferData.isEmpty() ) {
        return;
      }
      var gl = this.gl;

      var step = Float32Array.BYTES_PER_ELEMENT;
      var total = 3 + 4 + 3 + 3;
      var stride = step * total;

      this.shaderProgram.use();

      //TODO: Only call this when the canvas changes size
      //TODO: This backing scale multiply seems very buggy and contradicts everything we know!
      // Still, it gives the right behavior on iPad3 and OSX (non-retina).  Should be discussed and investigated.
      gl.uniform2f( this.shaderProgram.uniformLocations.uResolution, this.canvas.width / this.backingScale, this.canvas.height / this.backingScale );

      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );
      gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aPosition, 3, gl.FLOAT, false, stride, 0 );
      gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aVertexColor, 4, gl.FLOAT, false, stride, step * (3) );
      gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aTransform1, 3, gl.FLOAT, false, stride, step * (3 + 4) );
      gl.vertexAttribPointer( this.shaderProgram.attributeLocations.aTransform2, 3, gl.FLOAT, false, stride, step * (3 + 4 + 3) );

      gl.drawArrays( gl.TRIANGLES, 0, this.colorTriangleBufferData.vertexArray.length / 13 );

      this.shaderProgram.unuse();
    },

    bindVertexBuffer: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      // Keep track of the vertexArray for updating sublists of it
      this.vertexArray = new Float32Array( this.colorTriangleBufferData.vertexArray );
      gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW );
    },

    /**
     * Alternative to calling bufferSubData--just send the entire vertex buffer again.
     * Not clear when this may be a better alternative than using bufferSubData.
     */
    reBufferData: function() {
      var gl = this.gl;
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      // Keep track of the vertexArray for updating sublists of it
      gl.bufferData( gl.ARRAY_BUFFER, this.vertexArray, gl.DYNAMIC_DRAW );
    },

    updateTriangleBuffer: function( geometry ) {
      var gl = this.gl;

      // Update the vertex locations
      // Use a buffer view to only update the changed vertices
      // like //see http://stackoverflow.com/questions/19892022/webgl-optimizing-a-vertex-buffer-that-changes-values-vertex-count-every-frame
      // See also http://stackoverflow.com/questions/5497722/how-can-i-animate-an-object-in-webgl-modify-specific-vertices-not-full-transfor
      gl.bindBuffer( gl.ARRAY_BUFFER, this.vertexBuffer );

      //Update the Float32Array values
      for ( var i = geometry.startIndex; i < geometry.endIndex; i++ ) {
        this.vertexArray[ i ] = this.colorTriangleBufferData.vertexArray[ i ];
      }

      // Isolate the subarray of changed values
      var subArray = this.vertexArray.subarray( geometry.startIndex, geometry.endIndex );

      // Send new values to the GPU
      // See https://www.khronos.org/webgl/public-mailing-list/archives/1201/msg00110.html
      // The the offset is the index times the bytes per value
      gl.bufferSubData( gl.ARRAY_BUFFER, geometry.startIndex * 4, subArray );
    }
  } );
} );