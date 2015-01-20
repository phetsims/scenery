//  Copyright 2002-2014, University of Colorado Boulder

/**
 * Auxiliary functions designed to be used like a library rather than a framework, to simplify building
 * WebGL renderers.  (Used as a library means it should be very easy to skip using portions of this library where
 * necessary).
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );

  /**
   *
   * @constructor
   */
  function WebGLUtil() {
  }

  return inherit( Object, WebGLUtil, {}, {

    toShader: function( gl, source, type, typeString ) {
      var shader = gl.createShader( type );
      gl.shaderSource( shader, source );
      gl.compileShader( shader );
      if ( !gl.getShaderParameter( shader, gl.COMPILE_STATUS ) ) {
        console.log( 'ERROR IN ' + typeString + ' SHADER : ' + gl.getShaderInfoLog( shader ) );
        return false;
      }
      return shader;
    }
  } );
} );