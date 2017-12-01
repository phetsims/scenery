// Copyright 2017, University of Colorado Boulder

/**
 *
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );

  function snapshotToCanvas( snapshot ) {

    var canvas = document.createElement( 'canvas' );
    canvas.width = snapshot.width;
    canvas.height = snapshot.height;
    var context = canvas.getContext( '2d' );
    context.putImageData( snapshot, 0, 0 );
    $( canvas ).css( 'border', '1px solid black' );
    return canvas;
  }

  // TODO: factor out
// compares two pixel snapshots {ImageData} and uses the qunit's assert to verify they are the same
  function snapshotEquals( assert, a, b, threshold, message, extraDom ) {

    var isEqual = a.width === b.width && a.height === b.height;
    var largestDifference = 0;
    var totalDifference = 0;
    var colorDiffData = document.createElement( 'canvas' ).getContext( '2d' ).createImageData( a.width, a.height );
    var alphaDiffData = document.createElement( 'canvas' ).getContext( '2d' ).createImageData( a.width, a.height );
    if ( isEqual ) {
      for ( var i = 0; i < a.data.length; i++ ) {
        var diff = Math.abs( a.data[ i ] - b.data[ i ] );
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
        var alphaIndex = ( i - ( i % 4 ) + 3 );
        // grab the associated alpha channel and multiply it times the diff
        var alphaMultipliedDiff = ( i % 4 === 3 ) ? diff : diff * ( a.data[ alphaIndex ] / 255 ) * ( b.data[ alphaIndex ] / 255 );

        totalDifference += alphaMultipliedDiff;
        // if ( alphaMultipliedDiff > threshold ) {
        // console.log( message + ': ' + Math.abs( a.data[i] - b.data[i] ) );
        largestDifference = Math.max( largestDifference, alphaMultipliedDiff );
        // isEqual = false;
        // break;
        // }
      }
    }
    var averageDifference = totalDifference / ( 4 * a.width * a.height );
    if ( averageDifference > threshold ) {
      var display = $( '#display' );
      // header
      var note = document.createElement( 'h2' );
      $( note ).text( message );
      display.append( note );
      var differenceDiv = document.createElement( 'div' );
      $( differenceDiv ).text( '(actual) (expected) (color diff) (alpha diff) Diffs max: ' + largestDifference + ', average: ' + averageDifference );
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

  return snapshotEquals;
} );