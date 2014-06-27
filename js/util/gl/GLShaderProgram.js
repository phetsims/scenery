// Copyright 2002-2014, University of Colorado

/**
 * Shader program wrapper, so we can seamlessly recreate them on context loss.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  /*
   * @param gl {WebGLRenderingContext}
   * @param shaders {Array[GLShader]}
   */
  var GLShaderProgram = scenery.GLShaderProgram = function GLShaderProgram( gl, shaders, attributes, uniforms ) {
    var that = this;

    this._shaders = shaders;
    this._shaderProgram = GLShaderProgram.createProgram( gl, shaders );
    this._attributes = attributes;
    this._uniforms = uniforms;

    gl.useProgram( this._shaderProgram );

    _.each( attributes, function( attribute ) {
      var location = gl.getAttribLocation( that._shaderProgram, attribute );
      that[attribute] = location;
      gl.enableVertexAttribArray( location ); // TODO: what about where we don't always want them defined?
    } );

    _.each( uniforms, function( uniform ) {
      that[uniform] = gl.getUniformLocation( that._shaderProgram, uniform );
    } );
  };

  GLShaderProgram.prototype = {
    constructor: GLShaderProgram,

    use: function( gl ) {
      gl.useProgram( this._shaderProgram );
    }
  };

  GLShaderProgram.createProgram = function( gl, shaders ) {
    var shaderProgram = gl.createProgram();
    _.each( shaders, function( shader ) {
      gl.attachShader( shaderProgram, shader._shader );
    } );
    gl.linkProgram( shaderProgram );

    if ( !gl.getProgramParameter( shaderProgram, gl.LINK_STATUS ) ) {
      throw new Error( 'Could not initialise shaders' );
    }

    return shaderProgram;
  };

  return GLShaderProgram;
} );
