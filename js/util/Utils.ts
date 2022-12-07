// Copyright 2013-2022, University of Colorado Boulder

/**
 * General utility functions for Scenery
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Transform3 from '../../../dot/js/Transform3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import platform from '../../../phet-core/js/platform.js';
import { Features, scenery } from '../imports.js';

// convenience function
function p( x: number, y: number ): Vector2 {
  return new Vector2( x, y );
}

// TODO: remove flag and tests after we're done
const debugChromeBoundsScanning = false;

// detect properly prefixed transform and transformOrigin properties
const transformProperty = Features.transform;
const transformOriginProperty = Features.transformOrigin || 'transformOrigin'; // fallback, so we don't try to set an empty string property later

// Scenery applications that do not use WebGL may trigger a ~ 0.5 second pause shortly after launch on some platforms.
// Webgl is enabled by default but may be shut off for applications that know they will not want to use it
// see https://github.com/phetsims/scenery/issues/621
let webglEnabled = true;

let _extensionlessWebGLSupport: boolean | undefined; // lazily computed

const Utils = {
  /*---------------------------------------------------------------------------*
   * Transformation Utilities (TODO: separate file)
   *---------------------------------------------------------------------------*/

  /**
   * Prepares a DOM element for use with applyPreparedTransform(). Applies some CSS styles that are required, but
   * that we don't want to set while animating.
   */
  prepareForTransform( element: HTMLElement | SVGElement ): void {
    element.style[ transformOriginProperty ] = 'top left';
  },

  /**
   * Applies the CSS transform of the matrix to the element, with optional forcing of acceleration.
   * NOTE: prepareForTransform should be called at least once on the element before this method is used.
   */
  applyPreparedTransform( matrix: Matrix3, element: HTMLElement | SVGElement ): void {
    // NOTE: not applying translateZ, see http://stackoverflow.com/questions/10014461/why-does-enabling-hardware-acceleration-in-css3-slow-down-performance
    element.style[ transformProperty ] = matrix.getCSSTransform();
  },

  /**
   * Applies a CSS transform value string to a DOM element.
   * NOTE: prepareForTransform should be called at least once on the element before this method is used.
   */
  setTransform( transformString: string, element: HTMLElement | SVGElement ): void {
    element.style[ transformProperty ] = transformString;
  },

  /**
   * Removes a CSS transform from a DOM element.
   */
  unsetTransform( element: HTMLElement | SVGElement ): void {
    element.style[ transformProperty ] = '';
  },

  /**
   * Ensures that window.requestAnimationFrame and window.cancelAnimationFrame use a native implementation if possible,
   * otherwise using a simple setTimeout internally. See https://github.com/phetsims/scenery/issues/426
   */
  polyfillRequestAnimationFrame(): void {
    if ( !window.requestAnimationFrame || !window.cancelAnimationFrame ) {
      // Fallback implementation if no prefixed version is available
      if ( !Features.requestAnimationFrame || !Features.cancelAnimationFrame ) {
        window.requestAnimationFrame = callback => {
          const timeAtStart = Date.now();

          // NOTE: We don't want to rely on a common timer, so we're using the built-in form on purpose.
          return window.setTimeout( () => { // eslint-disable-line bad-sim-text
            callback( Date.now() - timeAtStart );
          }, 16 );
        };
        window.cancelAnimationFrame = clearTimeout;
      }
      // Fill in the non-prefixed names with the prefixed versions
      else {
        // @ts-expect-error
        window.requestAnimationFrame = window[ Features.requestAnimationFrame ];
        // @ts-expect-error
        window.cancelAnimationFrame = window[ Features.cancelAnimationFrame ];
      }
    }
  },

  /**
   * Returns the relative size of the context's backing store compared to the actual Canvas. For example, if it's 2,
   * the backing store has 2x2 the amount of pixels (4 times total).
   *
   * @returns The backing store pixel ratio.
   */
  backingStorePixelRatio( context: CanvasRenderingContext2D | WebGLRenderingContext ): number {
    // @ts-expect-error
    return context.webkitBackingStorePixelRatio ||
           // @ts-expect-error
           context.mozBackingStorePixelRatio ||
           // @ts-expect-error
           context.msBackingStorePixelRatio ||
           // @ts-expect-error
           context.oBackingStorePixelRatio ||
           // @ts-expect-error
           context.backingStorePixelRatio || 1;
  },

  /**
   * Returns the scaling factor that needs to be applied for handling a HiDPI Canvas
   * See see http://developer.apple.com/library/safari/#documentation/AudioVideo/Conceptual/HTML-canvas-guide/SettingUptheCanvas/SettingUptheCanvas.html#//apple_ref/doc/uid/TP40010542-CH2-SW5
   * And it's updated based on http://www.html5rocks.com/en/tutorials/canvas/hidpi/
   */
  backingScale( context: CanvasRenderingContext2D | WebGLRenderingContext ): number {
    if ( 'devicePixelRatio' in window ) {
      const backingStoreRatio = Utils.backingStorePixelRatio( context );

      return window.devicePixelRatio / backingStoreRatio;
    }
    return 1;
  },

  /**
   * Whether the native Canvas HTML5 API supports the 'filter' attribute (similar to the CSS/SVG filter attribute).
   */
  supportsNativeCanvasFilter(): boolean {
    return !!Features.canvasFilter;
  },

  /**
   * Whether we can handle arbitrary filters in Canvas by manipulating the ImageData returned. If we have a backing
   * store pixel ratio that is non-1, we'll be blurring out things during that operation, which would be unacceptable.
   */
  supportsImageDataCanvasFilter(): boolean {
    // @ts-expect-error TODO: scenery and typing
    return Utils.backingStorePixelRatio( scenery.scratchContext ) === 1;
  },

  /*---------------------------------------------------------------------------*
   * Text bounds utilities (TODO: separate file)
   *---------------------------------------------------------------------------*/

  /**
   * Given a data snapshot and transform, calculate range on how large / small the bounds can be. It's
   * very conservative, with an effective 1px extra range to allow for differences in anti-aliasing
   * for performance concerns, this does not support skews / rotations / anything but translation and scaling
   */
  scanBounds( imageData: ImageData, resolution: number, transform: Transform3 ): { minBounds: Bounds2; maxBounds: Bounds2 } {

    // entry will be true if any pixel with the given x or y value is non-rgba(0,0,0,0)
    const dirtyX = _.map( _.range( resolution ), () => false );
    const dirtyY = _.map( _.range( resolution ), () => false );

    for ( let x = 0; x < resolution; x++ ) {
      for ( let y = 0; y < resolution; y++ ) {
        const offset = 4 * ( y * resolution + x );
        if ( imageData.data[ offset ] !== 0 || imageData.data[ offset + 1 ] !== 0 || imageData.data[ offset + 2 ] !== 0 || imageData.data[ offset + 3 ] !== 0 ) {
          dirtyX[ x ] = true;
          dirtyY[ y ] = true;
        }
      }
    }

    const minX = _.indexOf( dirtyX, true );
    const maxX = _.lastIndexOf( dirtyX, true );
    const minY = _.indexOf( dirtyY, true );
    const maxY = _.lastIndexOf( dirtyY, true );

    // based on pixel boundaries. for minBounds, the inner edge of the dirty pixel. for maxBounds, the outer edge of the adjacent non-dirty pixel
    // results in a spread of 2 for the identity transform (or any translated form)
    const extraSpread = resolution / 16; // is Chrome antialiasing really like this? dear god... TODO!!!
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

  /**
   * Measures accurate bounds of a function that draws things to a Canvas.
   */
  canvasAccurateBounds( renderToContext: ( context: CanvasRenderingContext2D ) => void, options?: { precision?: number; resolution?: number; initialScale?: number } ): Bounds2 & { minBounds: Bounds2; maxBounds: Bounds2; isConsistent: boolean; precision: number } {
    // how close to the actual bounds do we need to be?
    const precision = ( options && options.precision ) ? options.precision : 0.001;

    // 512x512 default square resolution
    const resolution = ( options && options.resolution ) ? options.resolution : 128;

    // at 1/16x default, we want to be able to get the bounds accurately for something as large as 16x our initial resolution
    // divisible by 2 so hopefully we avoid more quirks from Canvas rendering engines
    const initialScale = ( options && options.initialScale ) ? options.initialScale : ( 1 / 16 );

    let minBounds = Bounds2.NOTHING;
    let maxBounds = Bounds2.EVERYTHING;

    const canvas = document.createElement( 'canvas' );
    canvas.width = resolution;
    canvas.height = resolution;
    const context = canvas.getContext( '2d' )!;

    if ( debugChromeBoundsScanning ) {
      $( window ).ready( () => {
        const header = document.createElement( 'h2' );
        $( header ).text( 'Bounds Scan' );
        $( '#display' ).append( header );
      } );
    }

    // TODO: Don't use Transform3 unless it is necessary
    function scan( transform: Transform3 ): { minBounds: Bounds2; maxBounds: Bounds2 } {
      // save/restore, in case the render tries to do any funny stuff like clipping, etc.
      context.save();
      transform.matrix.canvasSetTransform( context );
      renderToContext( context );
      context.restore();

      const data = context.getImageData( 0, 0, resolution, resolution );
      const minMaxBounds = Utils.scanBounds( data, resolution, transform );

      function snapshotToCanvas( snapshot: ImageData ): void {
        const canvas = document.createElement( 'canvas' );
        canvas.width = resolution;
        canvas.height = resolution;
        const context = canvas.getContext( '2d' )!;
        context.putImageData( snapshot, 0, 0 );
        $( canvas ).css( 'border', '1px solid black' );
        $( window ).ready( () => {
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
    function idealTransform( bounds: Bounds2 ): Transform3 {
      // so that the bounds-edge doesn't land squarely on the boundary
      const borderSize = 2;

      const scaleX = ( resolution - borderSize * 2 ) / ( bounds.maxX - bounds.minX );
      const scaleY = ( resolution - borderSize * 2 ) / ( bounds.maxY - bounds.minY );
      const translationX = -scaleX * bounds.minX + borderSize;
      const translationY = -scaleY * bounds.minY + borderSize;

      return new Transform3( Matrix3.translation( translationX, translationY ).timesMatrix( Matrix3.scaling( scaleX, scaleY ) ) );
    }

    const initialTransform = new Transform3();
    // make sure to initially center our object, so we don't miss the bounds
    initialTransform.append( Matrix3.translation( resolution / 2, resolution / 2 ) );
    initialTransform.append( Matrix3.scaling( initialScale ) );

    const coarseBounds = scan( initialTransform );

    minBounds = minBounds.union( coarseBounds.minBounds );
    maxBounds = maxBounds.intersection( coarseBounds.maxBounds );

    let tempMin;
    let tempMax;
    let refinedBounds;

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
          console.log( `transformed "min" minX: ${idealTransform( new Bounds2( maxBounds.minX, maxBounds.minY, minBounds.minX, maxBounds.maxY ) ).transformPosition2( p( minBounds.minX, 0 ) )}` );
          console.log( `transformed "max" minX: ${idealTransform( new Bounds2( maxBounds.minX, maxBounds.minY, minBounds.minX, maxBounds.maxY ) ).transformPosition2( p( maxBounds.minX, 0 ) )}` );
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
      console.log( `minBounds: ${minBounds}` );
      console.log( `maxBounds: ${maxBounds}` );
    }

    // @ts-expect-error
    const result: Bounds2 & { minBounds: Bounds2; maxBounds: Bounds2; isConsistent: boolean; precision: number } = new Bounds2(
      // Do finite checks so we don't return NaN
      ( isFinite( minBounds.minX ) && isFinite( maxBounds.minX ) ) ? ( minBounds.minX + maxBounds.minX ) / 2 : Number.POSITIVE_INFINITY,
      ( isFinite( minBounds.minY ) && isFinite( maxBounds.minY ) ) ? ( minBounds.minY + maxBounds.minY ) / 2 : Number.POSITIVE_INFINITY,
      ( isFinite( minBounds.maxX ) && isFinite( maxBounds.maxX ) ) ? ( minBounds.maxX + maxBounds.maxX ) / 2 : Number.NEGATIVE_INFINITY,
      ( isFinite( minBounds.maxY ) && isFinite( maxBounds.maxY ) ) ? ( minBounds.maxY + maxBounds.maxY ) / 2 : Number.NEGATIVE_INFINITY
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

  /*---------------------------------------------------------------------------*
   * WebGL utilities (TODO: separate file)
   *---------------------------------------------------------------------------*/

  /**
   * Finds the smallest power of 2 that is at least as large as n.
   *
   * @returns The smallest power of 2 that is greater than or equal n
   */
  toPowerOf2( n: number ): number {
    let result = 1;
    while ( result < n ) {
      result *= 2;
    }
    return result;
  },

  /**
   * Creates and compiles a GLSL Shader object in WebGL.
   */
  createShader( gl: WebGLRenderingContext, source: string, type: number ): WebGLShader {
    const shader = gl.createShader( type )!;
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

  applyWebGLContextDefaults( gl: WebGLRenderingContext ): void {
    // What color gets set when we call gl.clear()
    gl.clearColor( 0, 0, 0, 0 );

    // Blending similar to http://localhost/phet/git/webgl-blendfunctions/blendfuncseparate.html
    gl.enable( gl.BLEND );

    // NOTE: We switched back to a fully premultiplied setup, so we have the corresponding blend function.
    // For normal colors (and custom WebGLNode handling), it is necessary to use premultiplied values (multiplying the
    // RGB values by the alpha value for gl_FragColor). For textured triangles, it is assumed that the texture is
    // already premultiplied, so the built-in shader does not do the extra premultiplication.
    // See https://github.com/phetsims/energy-skate-park/issues/39, https://github.com/phetsims/scenery/issues/397
    // and https://stackoverflow.com/questions/39341564/webgl-how-to-correctly-blend-alpha-channel-png
    gl.blendFunc( gl.ONE, gl.ONE_MINUS_SRC_ALPHA );
  },

  /**
   * Set whether webgl should be enabled, see docs for webglEnabled
   */
  setWebGLEnabled( _webglEnabled: boolean ): void {
    webglEnabled = _webglEnabled;
  },

  /**
   * Check to see whether webgl is supported, using the same strategy as mrdoob and pixi.js
   *
   * @param [extensions] - A list of WebGL extensions that need to be supported
   */
  checkWebGLSupport( extensions?: string[] ): boolean {

    // The webgl check can be shut off, please see docs at webglEnabled declaration site
    if ( !webglEnabled ) {
      return false;
    }
    const canvas = document.createElement( 'canvas' );

    const args = { failIfMajorPerformanceCaveat: true };
    try {
      // @ts-expect-error
      const gl: WebGLRenderingContext | null = !!window.WebGLRenderingContext &&
                                               ( canvas.getContext( 'webgl', args ) || canvas.getContext( 'experimental-webgl', args ) );

      if ( !gl ) {
        return false;
      }

      if ( extensions ) {
        for ( let i = 0; i < extensions.length; i++ ) {
          if ( gl.getExtension( extensions[ i ] ) === null ) {
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

  /**
   * Check to see whether IE11 has proper clearStencil support (required for three.js to work well).
   */
  checkIE11StencilSupport(): boolean {
    const canvas = document.createElement( 'canvas' );

    try {
      // @ts-expect-error
      const gl: WebGLRenderingContext | null = !!window.WebGLRenderingContext &&
                                               ( canvas.getContext( 'webgl' ) || canvas.getContext( 'experimental-webgl' ) );

      if ( !gl ) {
        return false;
      }

      // Failure for https://github.com/mrdoob/three.js/issues/3600 / https://github.com/phetsims/molecule-shapes/issues/133
      gl.clearStencil( 0 );
      return gl.getError() === 0;
    }
    catch( e ) {
      return false;
    }
  },

  /**
   * Whether WebGL (with decent performance) is supported by the platform
   */
  get isWebGLSupported(): boolean {
    if ( _extensionlessWebGLSupport === undefined ) {
      _extensionlessWebGLSupport = Utils.checkWebGLSupport();
    }
    return _extensionlessWebGLSupport;
  },

  /**
   * Triggers a loss of a WebGL context, with a delayed restoration.
   *
   * NOTE: Only use this for debugging. Should not be called normally.
   */
  loseContext( gl: WebGLRenderingContext ): void {
    const extension = gl.getExtension( 'WEBGL_lose_context' );
    if ( extension ) {
      extension.loseContext();

      // NOTE: We don't want to rely on a common timer, so we're using the built-in form on purpose.
      setTimeout( () => { // eslint-disable-line bad-sim-text
        extension.restoreContext();
      }, 1000 );
    }
  },

  /**
   * Creates a string useful for working around https://github.com/phetsims/collision-lab/issues/177.
   */
  safariEmbeddingMarkWorkaround( str: string ): string {
    if ( platform.safari ) {
      // Add in zero-width spaces for Safari, so it doesn't have adjacent embedding marks ever (which seems to prevent
      // things).
      return str.split( '' ).join( '\u200B' );
    }
    else {
      return str;
    }
  }
};

scenery.register( 'Utils', Utils );
export default Utils;