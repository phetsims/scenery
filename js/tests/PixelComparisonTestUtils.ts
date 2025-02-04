// Copyright 2024-2025, University of Colorado Boulder

/**
 * Utility functions for pixel comparison tests.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import platform from '../../../phet-core/js/platform.js';
import Color from '../util/Color.js';
import Display from '../display/Display.js';
import Node from '../nodes/Node.js';
import type { RendererType } from '../nodes/Node.js';

const TESTED_RENDERERS: RendererType[] = [ 'canvas', 'svg', 'dom', 'webgl' ];

const PixelComparisonTestUtils = {

  DEFAULT_THRESHOLD: 1.5,

  /**
   * Returns an ImageData object representing pixel data from the provided image.
   */
  snapshotFromImage: ( image: HTMLImageElement ): ImageData => {

    const canvas = document.createElement( 'canvas' );
    canvas.width = image.width;
    canvas.height = image.height;
    const context = canvas.getContext( '2d' )!;
    context.drawImage( image, 0, 0, image.width, image.height );
    return context.getImageData( 0, 0, image.width, image.height );
  },

  /**
   * Draws the provided snapshot to a canvas and returns the canvas.
   */
  snapshotToCanvas: ( snapshot: ImageData ): HTMLCanvasElement => {
    const canvas = document.createElement( 'canvas' );
    canvas.width = snapshot.width;
    canvas.height = snapshot.height;
    const context = canvas.getContext( '2d' )!;
    context.putImageData( snapshot, 0, 0 );
    $( canvas ).css( 'border', '1px solid black' );
    return canvas;
  },

  /**
   * Checks to see if pixel comparison tests are supported on the current platform. Pixel comparisons
   * are only guaranteed on Firefox and Chrome.
   *
   * Returns true if supported, and logs an error message if not.
   */
  platformSupportsPixelComparisonTests: (): boolean => {
    const supported = platform.firefox || platform.chromium;
    if ( !supported ) {
      window.console && window.console.log && window.console.log( 'Not running pixel-comparison tests, platform not supported.' );
    }

    return supported;
  },

  /**
   * Compares two pixel snapshots {ImageData} and uses QUnit's assert to verify they are the same.
   */
  snapshotEquals: ( assert: Assert, a: ImageData, b: ImageData, threshold: number, message: string, extraDom?: HTMLElement ): boolean => {

    let isEqual = a.width === b.width && a.height === b.height;
    let largestDifference = 0;
    let totalDifference = 0;
    const colorDiffData = document.createElement( 'canvas' ).getContext( '2d' )!.createImageData( a.width, a.height );
    const alphaDiffData = document.createElement( 'canvas' ).getContext( '2d' )!.createImageData( a.width, a.height );
    if ( isEqual ) {
      for ( let i = 0; i < a.data.length; i++ ) {
        const diff = Math.abs( a.data[ i ] - b.data[ i ] );
        if ( i % 4 === 3 ) {
          colorDiffData.data[ i ] = 255;
          alphaDiffData.data[ i ] = 255;
          alphaDiffData.data[ i - 3 ] = diff; // red
          alphaDiffData.data[ i - 2 ] = diff; // green
          alphaDiffData.data[ i - 1 ] = diff; // blue
        }
        else {
          colorDiffData.data[ i ] = diff;
        }
        const alphaIndex = ( i - ( i % 4 ) + 3 );
        // grab the associated alpha channel and multiply it times the diff
        const alphaMultipliedDiff = ( i % 4 === 3 ) ? diff : diff * ( a.data[ alphaIndex ] / 255 ) * ( b.data[ alphaIndex ] / 255 );

        totalDifference += alphaMultipliedDiff;
        // if ( alphaMultipliedDiff > threshold ) {
        // console.log( message + ': ' + Math.abs( a.data[i] - b.data[i] ) );
        largestDifference = Math.max( largestDifference, alphaMultipliedDiff );
        // isEqual = false;
        // break;
        // }
      }
    }
    const averageDifference = totalDifference / ( 4 * a.width * a.height );
    if ( averageDifference > threshold ) {
      const display = $( '#display' );
      // header
      const note = document.createElement( 'h2' );
      $( note ).text( message );
      display.append( note );
      const differenceDiv = document.createElement( 'div' );
      $( differenceDiv ).text( `(actual) (expected) (color diff) (alpha diff) Diffs max: ${largestDifference}, average: ${averageDifference}` );
      display.append( differenceDiv );

      display.append( PixelComparisonTestUtils.snapshotToCanvas( a ) );
      display.append( PixelComparisonTestUtils.snapshotToCanvas( b ) );
      display.append( PixelComparisonTestUtils.snapshotToCanvas( colorDiffData ) );
      display.append( PixelComparisonTestUtils.snapshotToCanvas( alphaDiffData ) );

      if ( extraDom ) {
        display.append( extraDom );
      }

      // for a line-break
      display.append( document.createElement( 'div' ) );

      isEqual = false;
    }
    assert.ok( isEqual, message );
    return isEqual;
  },

  /**
   * Runs a pixel comparison test with QUnit between a reference data URL and a Display (with options and setup).
   *
   * @param name - Test name
   * @param setup - Called to set up the scene and display with rendered content.
   *                           function( scene, display, asyncCallback ). If asynchronous, call the asyncCallback when
   *                           the Display is ready to be rasterized.
   * @param dataURL - The reference data URL to compare against
   * @param threshold - Numerical threshold to determine how much error is acceptable
   * @param isAsync - Whether the setup function is asynchronous
   */
  pixelTest: (
    name: string,
    setup: ( scene: Node, display: Display, asyncCallback: () => void ) => void,
    dataURL: string,
    threshold: number,
    isAsync: boolean
  ): void => {

    QUnit.test( name, assert => {
      const done = assert.async();
      // set up the scene/display
      const scene = new Node();
      const display = new Display( scene, {
        preserveDrawingBuffer: true
      } );

      function loadImages(): void {
        // called when both images have been loaded
        function compareSnapshots(): void {
          const referenceSnapshot = PixelComparisonTestUtils.snapshotFromImage( referenceImage );
          const freshSnapshot = PixelComparisonTestUtils.snapshotFromImage( freshImage );

          display.domElement.style.position = 'relative'; // don't have it be absolutely positioned
          display.domElement.style.border = '1px solid black'; // border

          // the actual comparison statement
          PixelComparisonTestUtils.snapshotEquals( assert, freshSnapshot, referenceSnapshot, threshold, name, display.domElement );

          // tell qunit that we're done? (that's what the old version did, seems potentially wrong but working?)
          done();
        }

        // load images to compare
        let loadedCount = 0;
        const referenceImage = document.createElement( 'img' );
        const freshImage = document.createElement( 'img' );
        referenceImage.onload = freshImage.onload = () => {
          if ( ++loadedCount === 2 ) {
            compareSnapshots();
          }
        };
        referenceImage.onerror = () => {
          assert.ok( false, `${name} reference image failed to load` );
          done();
        };
        freshImage.onerror = () => {
          assert.ok( false, `${name} fresh image failed to load` );
          done();
        };

        referenceImage.src = dataURL;

        display.foreignObjectRasterization( url => {
          if ( !url ) {
            assert.ok( false, `${name} failed to rasterize the display` );
            done();
            return;
          }
          freshImage.src = url;
        } );
      }

      setup( scene, display, loadImages );

      if ( !isAsync ) {
        loadImages();
      }
    } );
  },

  /**
   * Like pixelTest, but for multiple listeners ({string[]}). Don't override the renderer on the scene.
   */
  multipleRendererTest: (
    name: string,
    setup: ( scene: Node, display: Display, asyncCallback: () => void ) => void,
    dataURL: string,
    threshold: number,
    renderers: RendererType[],
    isAsync?: boolean
  ): void => {
    for ( let i = 0; i < renderers.length; i++ ) {
      ( () => {
        const renderer = renderers[ i ];

        PixelComparisonTestUtils.pixelTest( `${name} (${renderer})`, ( scene, display, asyncCallback ) => {
          scene.renderer = renderer;
          setup( scene, display, asyncCallback );
        }, dataURL, threshold, !!isAsync );
      } )();
    }
  },

  COLORS: [
    new Color( 62, 171, 3 ),
    new Color( 23, 180, 77 ),
    new Color( 24, 183, 138 ),
    new Color( 23, 178, 194 ),
    new Color( 20, 163, 238 ),
    new Color( 71, 136, 255 ),
    new Color( 171, 101, 255 ),
    new Color( 228, 72, 235 ),
    new Color( 252, 66, 186 ),
    new Color( 252, 82, 127 )
  ],

  TESTED_RENDERERS: TESTED_RENDERERS,
  LAYOUT_TESTED_RENDERERS: [ 'svg' ] as RendererType[],
  NON_DOM_WEBGL_TESTED_RENDERERS: TESTED_RENDERERS.filter( renderer => renderer !== 'dom' && renderer !== 'webgl' )
};

export default PixelComparisonTestUtils;