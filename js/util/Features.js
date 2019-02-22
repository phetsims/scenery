// Copyright 2013-2015, University of Colorado Boulder


/**
 * Feature detection
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var detectPrefix = require( 'PHET_CORE/detectPrefix' );
  var scenery = require( 'SCENERY/scenery' );

  var Features = {};
  scenery.register( 'Features', Features );

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
      // var pngFallback = 'data:image/png';

      return url.slice( 0, target.length ) === target;
    }
    catch( e ) {
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
        canvas.toDataURL();
        Features[ name ] = true;
      }
      catch( e ) {
        Features[ name ] = false;
      }
    };
    img.onload = loadCall;
    try {
      img.src = black1x1Url;
      if ( img.complete ) {
        loadCall();
      }
    }
    catch( e ) {
      Features[ name ] = false;
    }
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
  var canvas = document.createElement( 'canvas' );
  var ctx = canvas.getContext( '2d' );
  Features.toDataURLHD = detectPrefix( canvas, 'toDataURLHD' );
  Features.createImageDataHD = detectPrefix( ctx, 'createImageDataHD' );
  Features.getImageDataHD = detectPrefix( ctx, 'getImageDataHD' );
  Features.putImageDataHD = detectPrefix( ctx, 'putImageDataHD' );
  Features.currentTransform = detectPrefix( ctx, 'currentTransform' );

  var span = document.createElement( 'span' );
  var div = document.createElement( 'div' );
  Features.textStroke = detectPrefix( span.style, 'textStroke' );
  Features.textStrokeColor = detectPrefix( span.style, 'textStrokeColor' );
  Features.textStrokeWidth = detectPrefix( span.style, 'textStrokeWidth' );

  Features.transform = detectPrefix( div.style, 'transform' );
  Features.transformOrigin = detectPrefix( div.style, 'transformOrigin' );
  Features.backfaceVisibility = detectPrefix( div.style, 'backfaceVisibility' );
  Features.borderRadius = detectPrefix( div.style, 'borderRadius' );

  Features.userSelect = detectPrefix( div.style, 'userSelect' );
  Features.touchAction = detectPrefix( div.style, 'touchAction' );
  Features.touchCallout = detectPrefix( div.style, 'touchCallout' );
  Features.userDrag = detectPrefix( div.style, 'userDrag' );
  Features.tapHighlightColor = detectPrefix( div.style, 'tapHighlightColor' );

  Features.fontSmoothing = detectPrefix( div.style, 'fontSmoothing' );

  Features.requestAnimationFrame = detectPrefix( window, 'requestAnimationFrame' );
  Features.cancelAnimationFrame = detectPrefix( window, 'cancelAnimationFrame' );

  // e.g. Features.setStyle( domElement, Features.transform, '...' ), and doesn't set it if no 'transform' attribute (prefixed or no) is found
  Features.setStyle = function( domElement, optionalKey, value ) {
    if ( optionalKey !== undefined ) {
      domElement.style[ optionalKey ] = value;
    }
  };

  // Whether passive is a supported option for adding event listeners,
  // see https://developer.mozilla.org/en-US/docs/Web/API/EventTarget/addEventListener#Improving_scrolling_performance_with_passive_listeners
  Features.passive = false;
  window.addEventListener( 'test', null, Object.defineProperty( {}, 'passive', {
    get: function() { // eslint-disable-line getter-return
      Features.passive = true;
    }
  } ) );

  return Features;
} );
