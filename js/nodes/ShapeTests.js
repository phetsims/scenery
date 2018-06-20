// Copyright 2017, University of Colorado Boulder

/**
 * Shape tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var LineStyles = require( 'KITE/util/LineStyles' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var snapshotEquals = require( 'SCENERY/tests/snapshotEquals' );
  var Vector2 = require( 'DOT/Vector2' );

  QUnit.module( 'Shape' );

  var canvasWidth = 320;
  var canvasHeight = 240;

  // takes a snapshot of a scene and stores the pixel data, so that we can compare them
  function snapshot( scene, width, height ) {

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

  // TODO: factor out
  function sceneEquals( assert, constructionA, constructionB, message, threshold ) {

    if ( threshold === undefined ) {
      threshold = 0;
    }

    var sceneA = new Node();
    var sceneB = new Node();

    constructionA( sceneA );
    constructionB( sceneB );

    // sceneA.renderScene();
    // sceneB.renderScene();

    var isEqual = snapshotEquals( assert, snapshot( sceneA ), snapshot( sceneB ), threshold, message );

    // TODO: consider showing if tests fail
    return isEqual;
  }

  // TODO: factor out
  function strokeEqualsFill( assert, shapeToStroke, shapeToFill, strokeNodeSetup, message ) { // eslint-disable-line no-unused-vars

    sceneEquals( assert, function( scene ) {
      var node = new Path( null );
      node.setShape( shapeToStroke );
      node.setStroke( '#000000' );
      if ( strokeNodeSetup ) { strokeNodeSetup( node ); }
      scene.addChild( node );
    }, function( scene ) {
      var node = new Path( null );
      node.setShape( shapeToFill );
      node.setFill( '#000000' );
      // node.setStroke( '#ff0000' ); // for debugging strokes
      scene.addChild( node );
      // node.validateBounds();
      // scene.addChild( new Path( {
      //   shape: kite.Shape.bounds( node.getSelfBounds() ),
      //   fill: 'rgba(0,0,255,0.5)'
      // } ) );
    }, message, 128 ); // threshold of 128 due to antialiasing differences between fill and stroke... :(
  }

  function p( x, y ) { return new Vector2( x, y ); }

  /* eslint-disable no-undef */

  QUnit.test( 'Verifying Line/Rect', function( assert ) {
    var lineWidth = 50;
    // /shapeToStroke, shapeToFill, strokeNodeSetup, message, debugFlag
    var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
    var fillShape = Shape.rectangle( 100, 100 - lineWidth / 2, 200, lineWidth );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineWidth( lineWidth ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Line Segment - butt', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 50;

    var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Line Segment - square', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 50;
    styles.lineCap = 'square';

    var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Line Segment - round', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 50;
    styles.lineCap = 'round';

    var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Line Join - Miter', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;
    styles.lineJoin = 'miter';

    var strokeShape = new Shape();
    strokeShape.moveTo( 70, 70 );
    strokeShape.lineTo( 140, 200 );
    strokeShape.lineTo( 210, 70 );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Line Join - Miter - Closed', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;
    styles.lineJoin = 'miter';

    var strokeShape = new Shape();
    strokeShape.moveTo( 70, 70 );
    strokeShape.lineTo( 140, 200 );
    strokeShape.lineTo( 210, 70 );
    strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Line Join - Round', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;
    styles.lineJoin = 'round';

    var strokeShape = new Shape();
    strokeShape.moveTo( 70, 70 );
    strokeShape.lineTo( 140, 200 );
    strokeShape.lineTo( 210, 70 );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Line Join - Round - Closed', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;
    styles.lineJoin = 'round';

    var strokeShape = new Shape();
    strokeShape.moveTo( 70, 70 );
    strokeShape.lineTo( 140, 200 );
    strokeShape.lineTo( 210, 70 );
    strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Line Join - Bevel - Closed', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;
    styles.lineJoin = 'bevel';

    var strokeShape = new Shape();
    strokeShape.moveTo( 70, 70 );
    strokeShape.lineTo( 140, 200 );
    strokeShape.lineTo( 210, 70 );
    strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Rect', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;

    var strokeShape = Shape.rectangle( 40, 40, 150, 150 );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Manual Rect', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;

    var strokeShape = new Shape();
    strokeShape.moveTo( 40, 40 );
    strokeShape.lineTo( 190, 40 );
    strokeShape.lineTo( 190, 190 );
    strokeShape.lineTo( 40, 190 );
    strokeShape.lineTo( 40, 40 );
    strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Hex', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;

    var strokeShape = Shape.regularPolygon( 6, 100 ).transformed( Matrix3.translation( 130, 130 ) );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Overlap', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;

    var strokeShape = new Shape();
    strokeShape.moveTo( 40, 40 );
    strokeShape.lineTo( 200, 200 );
    strokeShape.lineTo( 40, 200 );
    strokeShape.lineTo( 200, 40 );
    strokeShape.lineTo( 100, 0 );
    strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  var miterMagnitude = 160;
  var miterAnglesInDegrees = [ 5, 8, 10, 11.5, 13, 20, 24, 30, 45 ];

  _.each( miterAnglesInDegrees, function( miterAngle ) {
    var miterAngleRadians = miterAngle * Math.PI / 180;
    QUnit.test( 'Miter limit angle (degrees): ' + miterAngle + ' would change at ' + 1 / Math.sin( miterAngleRadians / 2 ), function( assert ) {
      var styles = new LineStyles();
      styles.lineWidth = 30;

      var strokeShape = new Shape();
      var point = new Vector2( 40, 100 );
      strokeShape.moveToPoint( point );
      point = point.plus( Vector2.X_UNIT.times( miterMagnitude ) );
      strokeShape.lineToPoint( point );
      point = point.plus( Vector2.createPolar( miterMagnitude, miterAngleRadians ).negated() );
      strokeShape.lineToPoint( point );
      var fillShape = strokeShape.getStrokedShape( styles );

      strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
  } );

  QUnit.test( 'Overlapping rectangles', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;

    var strokeShape = new Shape();
    strokeShape.rect( 40, 40, 100, 100 );
    strokeShape.rect( 50, 50, 100, 100 );
    strokeShape.rect( 80, 80, 100, 100 );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  QUnit.test( 'Bezier Offset', function( assert ) {
    var styles = new LineStyles();
    styles.lineWidth = 30;

    var strokeShape = new Shape();
    strokeShape.moveTo( 40, 40 );
    strokeShape.quadraticCurveTo( 100, 200, 160, 40 );
    // strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( assert, strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );
} );