// Copyright 2002-2013, University of Colorado

/**
 * Feature detection
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Features = scenery.Features = {};
  
  function lazy( name, computeValue ) {
    Features[name] = function() {
      var value = computeValue();
      Features[name] = function() {
        return value;
      };
      return value;
    };
  }
  
  function supportsDataURLFormatOutput( format ) {
    try {
      var canvas = document.createElement( 'canvas' );
      canvas.width = 1;
      canvas.height = 1;
      var context = canvas.getContext( '2d' );
      context.fillStyle = 'black';
      context.fillRect( 0, 0, 1, 1 );
      var url = canvas.toDataURL( [ format ] );
      
      var target = 'data:' + format;
      var pngFallback = 'data:image/png';
      
      return url.slice( 0, target.length ) === target;
    } catch ( e ) {
      return false;
    }
  }
  
  lazy( 'canvasPNGOutput', supportsDataURLFormatOutput.bind( window, 'image/png' ) );
  lazy( 'canvasJPEGOutput', supportsDataURLFormatOutput.bind( window, 'image/jpeg' ) );
  lazy( 'canvasGIFOutput', supportsDataURLFormatOutput.bind( window, 'image/gif' ) );
  lazy( 'canvasICONOutput', supportsDataURLFormatOutput.bind( window, 'image/x-icon' ) );
  
  return Features;
} );
