// Copyright 2002-2014, University of Colorado Boulder

var canvasWidth = 320;
var canvasHeight = 240;
//
// var unicodeTestStrings = [
//   'This is a test',
//   'Newline\nJaggies?',
//   '\u222b',
//   '\ufdfa',
//   '\u00a7',
//   '\u00C1',
//   '\u00FF',
//   '\u03A9',
//   '\u0906',
//   '\u79C1',
//   '\u9054',
//   'A\u030a\u0352\u0333\u0325\u0353\u035a\u035e\u035e',
//   '0\u0489',
//   '\u2588'
// ];

// takes a snapshot of a scene and stores the pixel data, so that we can compare them
function snapshot( scene, width, height ) {
  'use strict';
  
  width = width || canvasWidth;
  height = height || canvasHeight;

  var canvas = document.createElement( 'canvas' );
  canvas.width = width;
  canvas.height = height;
  var context = canvas.getContext( '2d' );
  scene.renderToCanvas( canvas, context );
  var data = context.getImageData( 0, 0, canvasWidth, canvasHeight );
  return data;
}

// function asyncSnapshot( scene, callback, width, height ) {
//   'use strict';
//
//   width = width || canvasWidth;
//   height = height || canvasHeight;
//
//   var canvas = document.createElement( 'canvas' );
//   canvas.width = width;
//   canvas.height = height;
//   var context = canvas.getContext( '2d' );
//   scene.renderToCanvas( canvas, context, function() {
//     var data = context.getImageData( 0, 0, width, height );
//     callback( data );
//   } );
// }

function snapshotToCanvas( snapshot ) {
  'use strict';
  
  var canvas = document.createElement( 'canvas' );
  canvas.width = snapshot.width;
  canvas.height = snapshot.height;
  var context = canvas.getContext( '2d' );
  context.putImageData( snapshot, 0, 0 );
  $( canvas ).css( 'border', '1px solid black' );
  return canvas;
}

// function imageFromDataURL( dataURL, callback ) {
//   'use strict';
//
//   var img = document.createElement( 'img' );
//
//   img.onload = function() {
//     callback( img );
//   };
//
//   img.src = dataURL;
// }

function snapshotFromImage( image ) { // eslint-disable-line no-unused-vars
  'use strict';
  
  var canvas = document.createElement( 'canvas' );
  canvas.width = image.width;
  canvas.height = image.height;
  var context = canvas.getContext( '2d' );
  context.drawImage( image, 0, 0, image.width, image.height );
  return context.getImageData( 0, 0, image.width, image.height );
}

// function snapshotFromDataURL( dataURL, callback ) {
//   'use strict';
//
//   imageFromDataURL( dataURL, function( image ) {
//     callback( snapshotFromImage( image ) );
//   } );
// }

// compares two pixel snapshots {ImageData} and uses the qunit's assert to verify they are the same
function snapshotEquals( a, b, threshold, message, extraDom ) {
  'use strict';
  
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
  ok( isEqual, message );
  return isEqual;
}

function sceneEquals( constructionA, constructionB, message, threshold ) {
  'use strict';
  
  if ( threshold === undefined ) {
    threshold = 0;
  }

  var sceneA = new scenery.Node();
  var sceneB = new scenery.Node();

  constructionA( sceneA );
  constructionB( sceneB );

  // sceneA.renderScene();
  // sceneB.renderScene();

  var isEqual = snapshotEquals( snapshot( sceneA ), snapshot( sceneB ), threshold, message );

  // TODO: consider showing if tests fail
  return isEqual;
}

function strokeEqualsFill( shapeToStroke, shapeToFill, strokeNodeSetup, message ) { // eslint-disable-line no-unused-vars
  'use strict';
  
  sceneEquals( function( scene ) {
    var node = new scenery.Path( null );
    node.setShape( shapeToStroke );
    node.setStroke( '#000000' );
    if ( strokeNodeSetup ) { strokeNodeSetup( node ); }
    scene.addChild( node );
  }, function( scene ) {
    var node = new scenery.Path( null );
    node.setShape( shapeToFill );
    node.setFill( '#000000' );
    // node.setStroke( '#ff0000' ); // for debugging strokes
    scene.addChild( node );
    // node.validateBounds();
    // scene.addChild( new scenery.Path( {
    //   shape: kite.Shape.bounds( node.getSelfBounds() ),
    //   fill: 'rgba(0,0,255,0.5)'
    // } ) );
  }, message, 128 ); // threshold of 128 due to antialiasing differences between fill and stroke... :(
}
//
// function compareShapeRenderers( shape, message ) {
//   'use strict';
//
//
// }
//
// function testTextBounds( getBoundsOfText, fontDrawingStyles, message ) {
//   'use strict';
//
//   var precision = 1;
//   var title = document.createElement( 'h2' );
//   $( title ).text( message );
//   $( '#display' ).append( title );
//   _.each( unicodeTestStrings, function( testString ) {
//     var testBounds = getBoundsOfText( testString, fontDrawingStyles );
//     var bestBounds = scenery.canvasTextBoundsAccurate( testString, fontDrawingStyles );
//
//     var widthOk = Math.abs( testBounds.getWidth() - bestBounds.getWidth() ) < precision;
//     var heightOk = Math.abs( testBounds.getHeight() - bestBounds.getHeight() ) < precision;
//     var xOk = Math.abs( testBounds.getX() - bestBounds.getX() ) < precision;
//     var yOk = Math.abs( testBounds.getY() - bestBounds.getY() ) < precision;
//
//     var allOk = widthOk && heightOk && xOk && yOk;
//
//     ok( widthOk, testString + ' width error: ' + Math.abs( testBounds.getWidth() - bestBounds.getWidth() ) );
//     ok( heightOk, testString + ' height error: ' + Math.abs( testBounds.getHeight() - bestBounds.getHeight() ) );
//     ok( xOk, testString + ' x error: ' + Math.abs( testBounds.getX() - bestBounds.getX() ) );
//     ok( yOk, testString + ' y error: ' + Math.abs( testBounds.getY() - bestBounds.getY() ) );
//
//     // show any failures
//     var pad = 5;
//     var scaling = 4; // scale it for display accuracy
//     var canvas = document.createElement( 'canvas' );
//     canvas.width = Math.ceil( bestBounds.getWidth() + pad * 2 ) * scaling;
//     canvas.height = Math.ceil( bestBounds.getHeight() + pad * 2 ) * scaling;
//     var context = canvas.getContext( '2d' );
//     context.scale( scaling, scaling );
//     context.translate( pad - bestBounds.getX(), pad - bestBounds.getY() ); // center the text in our bounds
//
//     // background bounds
//     context.fillStyle = allOk ? '#ccffcc' : '#ffcccc'; // red/green depending on whether it passed
//     context.fillRect( testBounds.getX(), testBounds.getY(), testBounds.getWidth(), testBounds.getHeight() );
//
//     // text on top
//     context.fillStyle = 'rgba(0,0,0,0.7)';
//     context.font = fontDrawingStyles.font;
//     context.textAlign = fontDrawingStyles.textAlign;
//     context.textBaseline = fontDrawingStyles.textBaseline;
//     context.direction = fontDrawingStyles.direction;
//     context.fillText( testString, 0, 0 );
//
//     $( canvas ).css( 'border', '1px solid black' );
//     $( '#display' ).append( canvas );
//   } );
//   throw new Error( 'deprecated, use accurateCanvasBounds instead' );
// }

function equalsApprox( a, b, message ) { // eslint-disable-line no-unused-vars
  'use strict';
  
  ok( Math.abs( a - b ) < 0.0000001, ( message ? message + ': ' : '' ) + a + ' =? ' + b );
}

function createTestNodeTree() { // eslint-disable-line no-unused-vars
  'use strict';
  
  var node = new scenery.Node();
  node.addChild( new scenery.Node() );
  node.addChild( new scenery.Node() );
  node.addChild( new scenery.Node() );

  node.children[ 0 ].addChild( new scenery.Node() );
  node.children[ 0 ].addChild( new scenery.Node() );
  node.children[ 0 ].addChild( new scenery.Node() );
  node.children[ 0 ].addChild( new scenery.Node() );
  node.children[ 0 ].addChild( new scenery.Node() );

  node.children[ 0 ].children[ 1 ].addChild( new scenery.Node() );
  node.children[ 0 ].children[ 3 ].addChild( new scenery.Node() );
  node.children[ 0 ].children[ 3 ].addChild( new scenery.Node() );

  node.children[ 0 ].children[ 3 ].children[ 0 ].addChild( new scenery.Node() );

  return node;
}
