// Copyright 2002-2014, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: Shapes' );

  var Shape = kite.Shape;

  function p( x, y ) { return new dot.Vector2( x, y ); }

  /* eslint-disable no-undef */

  test( 'Verifying Line/Rect', function() {
    var lineWidth = 50;
    // /shapeToStroke, shapeToFill, strokeNodeSetup, message, debugFlag
    var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
    var fillShape = Shape.rectangle( 100, 100 - lineWidth / 2, 200, lineWidth );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineWidth( lineWidth ); }, QUnit.config.current.testName );
  } );

  test( 'Line Segment - butt', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 50;

    var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Line Segment - square', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 50;
    styles.lineCap = 'square';

    var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Line Segment - round', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 50;
    styles.lineCap = 'round';

    var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Line Join - Miter', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;
    styles.lineJoin = 'miter';

    var strokeShape = new Shape();
    strokeShape.moveTo( 70, 70 );
    strokeShape.lineTo( 140, 200 );
    strokeShape.lineTo( 210, 70 );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Line Join - Miter - Closed', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;
    styles.lineJoin = 'miter';

    var strokeShape = new Shape();
    strokeShape.moveTo( 70, 70 );
    strokeShape.lineTo( 140, 200 );
    strokeShape.lineTo( 210, 70 );
    strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Line Join - Round', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;
    styles.lineJoin = 'round';

    var strokeShape = new Shape();
    strokeShape.moveTo( 70, 70 );
    strokeShape.lineTo( 140, 200 );
    strokeShape.lineTo( 210, 70 );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Line Join - Round - Closed', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;
    styles.lineJoin = 'round';

    var strokeShape = new Shape();
    strokeShape.moveTo( 70, 70 );
    strokeShape.lineTo( 140, 200 );
    strokeShape.lineTo( 210, 70 );
    strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Line Join - Bevel - Closed', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;
    styles.lineJoin = 'bevel';

    var strokeShape = new Shape();
    strokeShape.moveTo( 70, 70 );
    strokeShape.lineTo( 140, 200 );
    strokeShape.lineTo( 210, 70 );
    strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Rect', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;

    var strokeShape = Shape.rectangle( 40, 40, 150, 150 );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Manual Rect', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;

    var strokeShape = new Shape();
    strokeShape.moveTo( 40, 40 );
    strokeShape.lineTo( 190, 40 );
    strokeShape.lineTo( 190, 190 );
    strokeShape.lineTo( 40, 190 );
    strokeShape.lineTo( 40, 40 );
    strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Hex', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;

    var strokeShape = Shape.regularPolygon( 6, 100 ).transformed( dot.Matrix3.translation( 130, 130 ) );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Overlap', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;

    var strokeShape = new Shape();
    strokeShape.moveTo( 40, 40 );
    strokeShape.lineTo( 200, 200 );
    strokeShape.lineTo( 40, 200 );
    strokeShape.lineTo( 200, 40 );
    strokeShape.lineTo( 100, 0 );
    strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  var miterMagnitude = 160;
  var miterAnglesInDegrees = [ 5, 8, 10, 11.5, 13, 20, 24, 30, 45 ];

  _.each( miterAnglesInDegrees, function( miterAngle ) {
    var miterAngleRadians = miterAngle * Math.PI / 180;
    test( 'Miter limit angle (degrees): ' + miterAngle + ' would change at ' + 1 / Math.sin( miterAngleRadians / 2 ), function() {
      var styles = new kite.LineStyles();
      styles.lineWidth = 30;

      var strokeShape = new Shape();
      var point = new dot.Vector2( 40, 100 );
      strokeShape.moveToPoint( point );
      point = point.plus( dot.Vector2.X_UNIT.times( miterMagnitude ) );
      strokeShape.lineToPoint( point );
      point = point.plus( dot.Vector2.createPolar( miterMagnitude, miterAngleRadians ).negated() );
      strokeShape.lineToPoint( point );
      var fillShape = strokeShape.getStrokedShape( styles );

      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
  } );

  test( 'Overlapping rectangles', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;

    var strokeShape = new Shape();
    strokeShape.rect( 40, 40, 100, 100 );
    strokeShape.rect( 50, 50, 100, 100 );
    strokeShape.rect( 80, 80, 100, 100 );
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  test( 'Bezier Offset', function() {
    var styles = new kite.LineStyles();
    styles.lineWidth = 30;

    var strokeShape = new Shape();
    strokeShape.moveTo( 40, 40 );
    strokeShape.quadraticCurveTo( 100, 200, 160, 40 );
    // strokeShape.close();
    var fillShape = strokeShape.getStrokedShape( styles );

    strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
  } );

  /* eslint-enable */
})();
