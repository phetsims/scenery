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
  
  function supportsDataURLFormatOrigin( name, black1x1Url ) {
    var canvas = document.createElement( 'canvas' );
    canvas.width = 1;
    canvas.height = 1;
    var context = canvas.getContext( '2d' );
    
    var img = document.createElement( 'img' );
    img.crossOrigin = 'Anonymous'; // maybe setting the CORS attribute will help?
    
    var loadCall = function() {
      try {
        context.drawImage( img, 0, 0 );
        var url = canvas.toDataURL();
        Features[name] = true;
      } catch ( e ) {
        Features[name] = false;
      }
    };
    img.onload = loadCall;
    try {
      img.src = black1x1Url;
      if ( img.complete ) {
        loadCall();
      }
    } catch ( e ) {
      Features[name] = false;
    }
  }
  
  function prefixed( name ) {
    var result = [];
    result.push( name );
    
    // prepare for camel case
    name = name.charAt( 0 ).toUpperCase() + name.slice( 1 );
    
    // Chrome planning to not introduce prefixes in the future, hopefully we will be safe
    result.push( 'moz' + name );
    result.push( 'Moz' + name ); // some prefixes seem to have all-caps?
    result.push( 'webkit' + name );
    result.push( 'ms' + name );
    result.push( 'o' + name );
    
    return result;
  }
  
  function detect( obj, names ) {
    for ( var i = 0; i < names.length; i++ ) {
      if ( obj[names[i]] !== undefined ) {
        return names[i];
      }
    }
    return undefined;
  }
  
  Features.canvasPNGOutput = supportsDataURLFormatOutput( 'image/png' );
  Features.canvasJPEGOutput = supportsDataURLFormatOutput( 'image/jpeg' );
  Features.canvasGIFOutput = supportsDataURLFormatOutput( 'image/gif' );
  Features.canvasICONOutput = supportsDataURLFormatOutput( 'image/x-icon' );
  
  // 1x1 black output from Chrome Canvas in PNG
  supportsDataURLFormatOrigin( 'canvasPNGInput', 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIW2NkYGD4DwABCQEBtxmN7wAAAABJRU5ErkJggg==' );
  
  // 1x1 black output from Chrome Canvas in JPEG
  supportsDataURLFormatOrigin( 'canvasJPEGInput', 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD8qqKKKAP/2Q==' );
  
  /*
   * This is from the following SVG:
   *
   * <?xml version="1.0"?>
   * <svg version="1.1" xmlns="http://www.w3.org/2000/svg" viewport="0 0 1 1" width="1" height="1" >
   *   <rect x="0" y="0" width="1" height="1" rx="0" ry="0" style="fill: black; stroke: none;"></rect>
   * </svg>
   */
  supportsDataURLFormatOrigin( 'canvasSVGInput', 'data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIj8+DQo8c3ZnIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB2aWV3cG9ydD0iMCAwIDEgMSIgd2lkdGg9IjEiIGhlaWdodD0iMSIgPg0KICA8cmVjdCB4PSIwIiB5PSIwIiB3aWR0aD0iMSIgaGVpZ2h0PSIxIiByeD0iMCIgcnk9IjAiIHN0eWxlPSJmaWxsOiBibGFjazsgc3Ryb2tlOiBub25lOyI+PC9yZWN0Pg0KPC9zdmc+DQo=' );
  
  // 1x1 black output from Photoshop in GIF
  supportsDataURLFormatOrigin( 'canvasGIFInput', 'data:image/gif;base64,R0lGODlhAQABAJEAAAAAAP///////wAAACH5BAEAAAIALAAAAAABAAEAAAICRAEAOw==' );
  
  // canvas prefixed names
  var ctx = document.createElement( 'canvas' ).getContext( '2d' );
  Features.createImageDataHD = detect( ctx, prefixed( 'createImageDataHD' ) );
  Features.getImageDataHD = detect( ctx, prefixed( 'getImageDataHD' ) );
  Features.putImageDataHD = detect( ctx, prefixed( 'putImageDataHD' ) );
  
  var span = document.createElement( 'span' );
  var div = document.createElement( 'div' );
  Features.textStroke = detect( span.style, prefixed( 'textStroke' ) );
  Features.textStrokeColor = detect( span.style, prefixed( 'textStrokeColor' ) );
  Features.textStrokeWidth = detect( span.style, prefixed( 'textStrokeWidth' ) );
  
  Features.transform = detect( div.style, prefixed( 'transform' ) );
  Features.transformOrigin = detect( div.style, prefixed( 'transformOrigin' ) );
  Features.backfaceVisibility = detect( div.style, prefixed( 'backfaceVisibility' ) );
  Features.borderRadius = detect( div.style, prefixed( 'borderRadius' ) );
  
  return Features;
} );
