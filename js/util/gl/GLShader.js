// Copyright 2002-2013, University of Colorado

/**
 * Shader wrapper, so we can seamlessly recreate them on context loss.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  var GLShader = scenery.GLShader = function GLShader( gl, src, type ) {
    this._src = src;
    this._type = type;
    this._shader = GLShader.compileShader( gl, src, type );
  };
  
  GLShader.fragmentShader = function( gl, src ) {
    return new GLShader( gl, src, gl.FRAGMENT_SHADER );
  };
  
  GLShader.vertexShader = function( gl, src ) {
    return new GLShader( gl, src, gl.VERTEX_SHADER );
  };
  
  GLShader.compileShader = function( gl, src, type ) {
    var shader = gl.createShader( type );
    gl.shaderSource( shader, src );
    gl.compileShader( shader );
    
    if ( !gl.getShaderParameter( shader, gl.COMPILE_STATUS ) ) {
      throw new Error( gl.getShaderInfoLog( shader ) );
    }
    return shader;
  };
  
  return GLShader;
} );
