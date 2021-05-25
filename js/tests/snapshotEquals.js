// Copyright 2017-2021, University of Colorado Boulder

/**
 *
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import scenery from '../scenery.js';

function snapshotToCanvas( snapshot ) {

  const canvas = document.createElement( 'canvas' );
  canvas.width = snapshot.width;
  canvas.height = snapshot.height;
  const context = canvas.getContext( '2d' );
  context.putImageData( snapshot, 0, 0 );
  $( canvas ).css( 'border', '1px solid black' );
  return canvas;
}

// TODO: factor out
// compares two pixel snapshots {ImageData} and uses the qunit's assert to verify they are the same
function snapshotEquals( assert, a, b, threshold, message, extraDom ) {

  let isEqual = a.width === b.width && a.height === b.height;
  let largestDifference = 0;
  let totalDifference = 0;
  const colorDiffData = document.createElement( 'canvas' ).getContext( '2d' ).createImageData( a.width, a.height );
  const alphaDiffData = document.createElement( 'canvas' ).getContext( '2d' ).createImageData( a.width, a.height );
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

    display.append( snapshotToCanvas( a ) );
    display.append( snapshotToCanvas( b ) );
    display.append( snapshotToCanvas( colorDiffData ) );
    display.append( snapshotToCanvas( alphaDiffData ) );

    if ( extraDom ) {
      display.append( extraDom );
    }

    // for a line-break
    display.append( document.createElement( 'div' ) );

    isEqual = false;
  }
  assert.ok( isEqual, message );
  return isEqual;
}

scenery.register( 'snapshotEquals', snapshotEquals );

export default snapshotEquals;