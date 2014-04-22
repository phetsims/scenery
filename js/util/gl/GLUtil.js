// Copyright 2002-2014, University of Colorado

/**
 * General utility functions for WebGL
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  var GLUtil = scenery.GLUtil = {
    getWebGLContext: function( canvas ) {
      var gl = null;
      var contextNames = ['webgl', 'experimental-webgl', 'webkit-3d', 'moz-webgl'];
      for ( var i = 0; i < contextNames.length; i++ ) {
        try {
          gl = canvas.getContext( contextNames[i] );
        } catch ( e ) {
          // consider storing this failure somewhere?
        }
        if ( gl ) {
          break;
        }
      }
      
      return gl;
    }
  };
  
  return GLUtil;
} );
