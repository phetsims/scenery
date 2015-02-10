// Copyright 2002-2014, University of Colorado Boulder


/**
 * General utility functions for Scenery
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  var Matrix3 = require( 'DOT/Matrix3' );
  var Transform3 = require( 'DOT/Transform3' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Vector2 = require( 'DOT/Vector2' );
  var Features = require( 'SCENERY/util/Features' );

  // convenience function
  function p( x, y ) {
    return new Vector2( x, y );
  }

  // TODO: remove flag and tests after we're done
  var debugChromeBoundsScanning = false;

  // detect properly prefixed transform and transformOrigin properties
  var transformProperty = Features.transform;
  var transformOriginProperty = Features.transformOrigin || 'transformOrigin'; // fallback, so we don't try to set an empty string property later

  scenery.Util = {
    // like _.extend, but with hardcoded support for https://github.com/documentcloud/underscore/pull/986
    extend: function( obj ) {
      _.each( Array.prototype.slice.call( arguments, 1 ), function( source ) {
        if ( source ) {
          for ( var prop in source ) {
            Object.defineProperty( obj, prop, Object.getOwnPropertyDescriptor( source, prop ) );
          }
        }
      } );
      return obj;
    },

    // @deprecated (bad performance since it is setting multiple properties). see applyPreparedTransform
    applyCSSTransform: function( matrix, element, forceAcceleration ) {
      var transformCSS = matrix.getCSSTransform();
      // notes on triggering hardware acceleration: http://creativejs.com/2011/12/day-2-gpu-accelerate-your-dom-elements/

      if ( forceAcceleration ) {
        element.style.webkitBackfaceVisibility = 'hidden';
        transformCSS += ' translateZ(0)';
      }

      element.style[ transformProperty ] = transformCSS;
      element.style[ transformOriginProperty ] = 'top left'; //OHTWO TODO: performance: this only needs to be set once!
    },

    prepareForTransform: function( element, forceAcceleration ) {
      element.style[ transformOriginProperty ] = 'top left';
      if ( forceAcceleration ) {
        scenery.Util.setTransformAcceleration( element );
      }
      else {
        scenery.Util.unsetTransformAcceleration( element );
      }
    },

    setTransformAcceleration: function( element ) {
      element.style.webkitBackfaceVisibility = 'hidden';
    },

    unsetTransformAcceleration: function( element ) {
      element.style.webkitBackfaceVisibility = '';
    },

    // applies the CSS transform of the {Matrix3} matrix to the element, with optional forcing of acceleration. prepareForTransform should be called before this method
    // is used, and they should use the same parameter value for forceAcceleration
    applyPreparedTransform: function( matrix, element, forceAcceleration ) {
      // NOTE: not applying translateZ, see http://stackoverflow.com/questions/10014461/why-does-enabling-hardware-acceleration-in-css3-slow-down-performance
      element.style[ transformProperty ] = matrix.getCSSTransform();
    },

    setTransform: function( transformString, element, forceAcceleration ) {
      assert && assert( typeof transformString === 'string' );

      element.style[ transformProperty ] = transformString;
    },

    unsetTransform: function( element ) {
      element.style[ transformProperty ] = '';
    },

    testAssert: function() {
      return 'assert.basic: ' + ( assert ? 'true' : 'false' );
    },

    testAssertExtra: function() {
      return 'assert.slow: ' + ( assertSlow ? 'true' : 'false' );
    },

    /*---------------------------------------------------------------------------*
     * window.requestAnimationFrame polyfill, by Erik Moller (http://my.opera.com/emoller/blog/2011/12/20/requestanimationframe-for-smart-er-animating)
     * referenced by initial Paul Irish article at http://paulirish.com/2011/requestanimationframe-for-smart-animating/
     *----------------------------------------------------------------------------*/
    polyfillRequestAnimationFrame: function() {
      var lastTime = 0;
      var vendors = [ 'ms', 'moz', 'webkit', 'o' ];
      for ( var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x ) {
        window.requestAnimationFrame = window[ vendors[ x ] + 'RequestAnimationFrame' ];
        window.cancelAnimationFrame = window[ vendors[ x ] + 'CancelAnimationFrame' ] || window[ vendors[ x ] + 'CancelRequestAnimationFrame' ];
      }

      if ( !window.requestAnimationFrame ) {
        window.requestAnimationFrame = function( callback ) {
          var currTime = new Date().getTime();
          var timeToCall = Math.max( 0, 16 - (currTime - lastTime) );
          var id = window.setTimeout( function() { callback( currTime + timeToCall ); },
            timeToCall );
          lastTime = currTime + timeToCall;
          return id;
        };
      }

      if ( !window.cancelAnimationFrame ) {
        window.cancelAnimationFrame = function( id ) {
          clearTimeout( id );
        };
      }
    },

    backingStorePixelRatio: function( context ) {
      return context.webkitBackingStorePixelRatio ||
             context.mozBackingStorePixelRatio ||
             context.msBackingStorePixelRatio ||
             context.oBackingStorePixelRatio ||
             context.backingStorePixelRatio || 1;
    },

    // see http://developer.apple.com/library/safari/#documentation/AudioVideo/Conceptual/HTML-canvas-guide/SettingUptheCanvas/SettingUptheCanvas.html#//apple_ref/doc/uid/TP40010542-CH2-SW5
    // and updated based on http://www.html5rocks.com/en/tutorials/canvas/hidpi/
    backingScale: function( context ) {
      if ( 'devicePixelRatio' in window ) {
        var backingStoreRatio = Util.backingStorePixelRatio( context );

        return window.devicePixelRatio / backingStoreRatio;
      }
      return 1;
    },

    // given a data snapshot and transform, calculate range on how large / small the bounds can be
    // very conservative, with an effective 1px extra range to allow for differences in anti-aliasing
    // for performance concerns, this does not support skews / rotations / anything but translation and scaling
    scanBounds: function( imageData, resolution, transform ) {

      // entry will be true if any pixel with the given x or y value is non-rgba(0,0,0,0)
      var dirtyX = _.map( _.range( resolution ), function() { return false; } );
      var dirtyY = _.map( _.range( resolution ), function() { return false; } );

      for ( var x = 0; x < resolution; x++ ) {
        for ( var y = 0; y < resolution; y++ ) {
          var offset = 4 * ( y * resolution + x );
          if ( imageData.data[ offset ] !== 0 || imageData.data[ offset + 1 ] !== 0 || imageData.data[ offset + 2 ] !== 0 || imageData.data[ offset + 3 ] !== 0 ) {
            dirtyX[ x ] = true;
            dirtyY[ y ] = true;
          }
        }
      }

      var minX = _.indexOf( dirtyX, true );
      var maxX = _.lastIndexOf( dirtyX, true );
      var minY = _.indexOf( dirtyY, true );
      var maxY = _.lastIndexOf( dirtyY, true );

      // based on pixel boundaries. for minBounds, the inner edge of the dirty pixel. for maxBounds, the outer edge of the adjacent non-dirty pixel
      // results in a spread of 2 for the identity transform (or any translated form)
      var extraSpread = resolution / 16; // is Chrome antialiasing really like this? dear god... TODO!!!
      return {
        minBounds: new Bounds2(
          ( minX < 1 || minX >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( minX + 1 + extraSpread, 0 ) ).x,
          ( minY < 1 || minY >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( 0, minY + 1 + extraSpread ) ).y,
          ( maxX < 1 || maxX >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( maxX - extraSpread, 0 ) ).x,
          ( maxY < 1 || maxY >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( 0, maxY - extraSpread ) ).y
        ),
        maxBounds: new Bounds2(
          ( minX < 1 || minX >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( minX - 1 - extraSpread, 0 ) ).x,
          ( minY < 1 || minY >= resolution - 1 ) ? Number.NEGATIVE_INFINITY : transform.inversePosition2( p( 0, minY - 1 - extraSpread ) ).y,
          ( maxX < 1 || maxX >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( maxX + 2 + extraSpread, 0 ) ).x,
          ( maxY < 1 || maxY >= resolution - 1 ) ? Number.POSITIVE_INFINITY : transform.inversePosition2( p( 0, maxY + 2 + extraSpread ) ).y
        )
      };
    },

    canvasAccurateBounds: function( renderToContext, options ) {
      // how close to the actual bounds do we need to be?
      var precision = ( options && options.precision ) ? options.precision : 0.001;

      // 512x512 default square resolution
      var resolution = ( options && options.resolution ) ? options.resolution : 128;

      // at 1/16x default, we want to be able to get the bounds accurately for something as large as 16x our initial resolution
      // divisible by 2 so hopefully we avoid more quirks from Canvas rendering engines
      var initialScale = ( options && options.initialScale ) ? options.initialScale : ( 1 / 16 );

      var minBounds = Bounds2.NOTHING;
      var maxBounds = Bounds2.EVERYTHING;

      var canvas = document.createElement( 'canvas' );
      canvas.width = resolution;
      canvas.height = resolution;
      var context = canvas.getContext( '2d' );

      if ( debugChromeBoundsScanning ) {
        $( window ).ready( function() {
          var header = document.createElement( 'h2' );
          $( header ).text( 'Bounds Scan' );
          $( '#display' ).append( header );
        } );
      }

      function scan( transform ) {
        // save/restore, in case the render tries to do any funny stuff like clipping, etc.
        context.save();
        transform.matrix.canvasSetTransform( context );
        renderToContext( context );
        context.restore();

        var data = context.getImageData( 0, 0, resolution, resolution );
        var minMaxBounds = Util.scanBounds( data, resolution, transform );

        function snapshotToCanvas( snapshot ) {
          var canvas = document.createElement( 'canvas' );
          canvas.width = resolution;
          canvas.height = resolution;
          var context = canvas.getContext( '2d' );
          context.putImageData( snapshot, 0, 0 );
          $( canvas ).css( 'border', '1px solid black' );
          $( window ).ready( function() {
            //$( '#display' ).append( $( document.createElement( 'div' ) ).text( 'Bounds: ' +  ) );
            $( '#display' ).append( canvas );
          } );
        }

        // TODO: remove after debug
        if ( debugChromeBoundsScanning ) {
          snapshotToCanvas( data );
        }

        context.clearRect( 0, 0, resolution, resolution );

        return minMaxBounds;
      }

      // attempts to map the bounds specified to the entire testing canvas (minus a fine border), so we can nail down the location quickly
      function idealTransform( bounds ) {
        // so that the bounds-edge doesn't land squarely on the boundary
        var borderSize = 2;

        var scaleX = ( resolution - borderSize * 2 ) / ( bounds.maxX - bounds.minX );
        var scaleY = ( resolution - borderSize * 2 ) / ( bounds.maxY - bounds.minY );
        var translationX = -scaleX * bounds.minX + borderSize;
        var translationY = -scaleY * bounds.minY + borderSize;

        return new Transform3( Matrix3.translation( translationX, translationY ).timesMatrix( Matrix3.scaling( scaleX, scaleY ) ) );
      }

      var initialTransform = new Transform3();
      // make sure to initially center our object, so we don't miss the bounds
      initialTransform.append( Matrix3.translation( resolution / 2, resolution / 2 ) );
      initialTransform.append( Matrix3.scaling( initialScale ) );

      var coarseBounds = scan( initialTransform );

      minBounds = minBounds.union( coarseBounds.minBounds );
      maxBounds = maxBounds.intersection( coarseBounds.maxBounds );

      var tempMin, tempMax, refinedBounds;

      // minX
      tempMin = maxBounds.minY;
      tempMax = maxBounds.maxY;
      while ( isFinite( minBounds.minX ) && isFinite( maxBounds.minX ) && Math.abs( minBounds.minX - maxBounds.minX ) > precision ) {
        // use maximum bounds except for the x direction, so we don't miss things that we are looking for
        refinedBounds = scan( idealTransform( new Bounds2( maxBounds.minX, tempMin, minBounds.minX, tempMax ) ) );

        if ( minBounds.minX <= refinedBounds.minBounds.minX && maxBounds.minX >= refinedBounds.maxBounds.minX ) {
          // sanity check - break out of an infinite loop!
          if ( debugChromeBoundsScanning ) {
            console.log( 'warning, exiting infinite loop!' );
            console.log( 'transformed "min" minX: ' + idealTransform( new Bounds2( maxBounds.minX, maxBounds.minY, minBounds.minX, maxBounds.maxY ) ).transformPosition2( p( minBounds.minX, 0 ) ) );
            console.log( 'transformed "max" minX: ' + idealTransform( new Bounds2( maxBounds.minX, maxBounds.minY, minBounds.minX, maxBounds.maxY ) ).transformPosition2( p( maxBounds.minX, 0 ) ) );
          }
          break;
        }

        minBounds = minBounds.withMinX( Math.min( minBounds.minX, refinedBounds.minBounds.minX ) );
        maxBounds = maxBounds.withMinX( Math.max( maxBounds.minX, refinedBounds.maxBounds.minX ) );
        tempMin = Math.max( tempMin, refinedBounds.maxBounds.minY );
        tempMax = Math.min( tempMax, refinedBounds.maxBounds.maxY );
      }

      // maxX
      tempMin = maxBounds.minY;
      tempMax = maxBounds.maxY;
      while ( isFinite( minBounds.maxX ) && isFinite( maxBounds.maxX ) && Math.abs( minBounds.maxX - maxBounds.maxX ) > precision ) {
        // use maximum bounds except for the x direction, so we don't miss things that we are looking for
        refinedBounds = scan( idealTransform( new Bounds2( minBounds.maxX, tempMin, maxBounds.maxX, tempMax ) ) );

        if ( minBounds.maxX >= refinedBounds.minBounds.maxX && maxBounds.maxX <= refinedBounds.maxBounds.maxX ) {
          // sanity check - break out of an infinite loop!
          if ( debugChromeBoundsScanning ) {
            console.log( 'warning, exiting infinite loop!' );
          }
          break;
        }

        minBounds = minBounds.withMaxX( Math.max( minBounds.maxX, refinedBounds.minBounds.maxX ) );
        maxBounds = maxBounds.withMaxX( Math.min( maxBounds.maxX, refinedBounds.maxBounds.maxX ) );
        tempMin = Math.max( tempMin, refinedBounds.maxBounds.minY );
        tempMax = Math.min( tempMax, refinedBounds.maxBounds.maxY );
      }

      // minY
      tempMin = maxBounds.minX;
      tempMax = maxBounds.maxX;
      while ( isFinite( minBounds.minY ) && isFinite( maxBounds.minY ) && Math.abs( minBounds.minY - maxBounds.minY ) > precision ) {
        // use maximum bounds except for the y direction, so we don't miss things that we are looking for
        refinedBounds = scan( idealTransform( new Bounds2( tempMin, maxBounds.minY, tempMax, minBounds.minY ) ) );

        if ( minBounds.minY <= refinedBounds.minBounds.minY && maxBounds.minY >= refinedBounds.maxBounds.minY ) {
          // sanity check - break out of an infinite loop!
          if ( debugChromeBoundsScanning ) {
            console.log( 'warning, exiting infinite loop!' );
          }
          break;
        }

        minBounds = minBounds.withMinY( Math.min( minBounds.minY, refinedBounds.minBounds.minY ) );
        maxBounds = maxBounds.withMinY( Math.max( maxBounds.minY, refinedBounds.maxBounds.minY ) );
        tempMin = Math.max( tempMin, refinedBounds.maxBounds.minX );
        tempMax = Math.min( tempMax, refinedBounds.maxBounds.maxX );
      }

      // maxY
      tempMin = maxBounds.minX;
      tempMax = maxBounds.maxX;
      while ( isFinite( minBounds.maxY ) && isFinite( maxBounds.maxY ) && Math.abs( minBounds.maxY - maxBounds.maxY ) > precision ) {
        // use maximum bounds except for the y direction, so we don't miss things that we are looking for
        refinedBounds = scan( idealTransform( new Bounds2( tempMin, minBounds.maxY, tempMax, maxBounds.maxY ) ) );

        if ( minBounds.maxY >= refinedBounds.minBounds.maxY && maxBounds.maxY <= refinedBounds.maxBounds.maxY ) {
          // sanity check - break out of an infinite loop!
          if ( debugChromeBoundsScanning ) {
            console.log( 'warning, exiting infinite loop!' );
          }
          break;
        }

        minBounds = minBounds.withMaxY( Math.max( minBounds.maxY, refinedBounds.minBounds.maxY ) );
        maxBounds = maxBounds.withMaxY( Math.min( maxBounds.maxY, refinedBounds.maxBounds.maxY ) );
        tempMin = Math.max( tempMin, refinedBounds.maxBounds.minX );
        tempMax = Math.min( tempMax, refinedBounds.maxBounds.maxX );
      }

      if ( debugChromeBoundsScanning ) {
        console.log( 'minBounds: ' + minBounds );
        console.log( 'maxBounds: ' + maxBounds );
      }

      var result = new Bounds2(
        ( minBounds.minX + maxBounds.minX ) / 2,
        ( minBounds.minY + maxBounds.minY ) / 2,
        ( minBounds.maxX + maxBounds.maxX ) / 2,
        ( minBounds.maxY + maxBounds.maxY ) / 2
      );

      // extra data about our bounds
      result.minBounds = minBounds;
      result.maxBounds = maxBounds;
      result.isConsistent = maxBounds.containsBounds( minBounds );
      result.precision = Math.max(
        Math.abs( minBounds.minX - maxBounds.minX ),
        Math.abs( minBounds.minY - maxBounds.minY ),
        Math.abs( minBounds.maxX - maxBounds.maxX ),
        Math.abs( minBounds.maxY - maxBounds.maxY )
      );

      // return the average
      return result;
    },

    // returns the smallest power of 2 that is greater than or equal
    toPowerOf2: function( n ) {
      var result = 1;
      while ( result < n ) {
        result *= 2;
      }
      return result;
    },

    /*
     * @param type should be: gl.VERTEX_SHADER or gl.FRAGMENT_SHADER
     * @param source {String}, the shader source code.
     */
    createShader: function( gl, source, type ) {
      var shader = gl.createShader( type );
      gl.shaderSource( shader, source );
      gl.compileShader( shader );

      if ( !gl.getShaderParameter( shader, gl.COMPILE_STATUS ) ) {
        console.log( 'GLSL compile error:' );
        console.log( gl.getShaderInfoLog( shader ) );
        console.log( source );

        // Normally it would be best to throw an exception here, but a context loss could cause the shader parameter check
        // to fail, and we must handle context loss gracefully between any adjacent pair of gl calls.
        // Therefore, we simply report the errors to the console.  See #279
      }

      return shader;
    },

    /**
     * Check to see whether webgl is supported, using the same strategy as mrdoob and pixi.js
     *
     * @param {Array.<string>} [extensions] - A list of WebGL extensions that need to be supported
     */
    checkWebGLSupport: function( extensions ) {
      var canvas = document.createElement( 'canvas' );

      var args = { failIfMajorPerformanceCaveat: true };
      try {
        var gl =
          !!window.WebGLRenderingContext &&
          (canvas.getContext( 'webgl', args ) || canvas.getContext( 'experimental-webgl', args ));

        if ( !gl ) {
          return false;
        }

        if ( extensions ) {
          for ( var i = 0; i < extensions.length; i++ ) {
            if ( gl.getExtension( extensions[i] ) === null ) {
              return false;
            }
          }
        }

        return true;
      }
      catch( e ) {
        return false;
      }
    },

    // Whether WebGL (with decent performance) is supported by the platform
    get isWebGLSupported() {
      if ( this._extensionlessWebGLSupport === undefined ) {
        this._extensionlessWebGLSupport = scenery.Util.checkWebGLSupport();
      }
      return this._extensionlessWebGLSupport;
    }
  };
  var Util = scenery.Util;

  return Util;
} );
