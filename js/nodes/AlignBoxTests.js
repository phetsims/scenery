// Copyright 2017, University of Colorado Boulder

/**
 * AlignBox tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var AlignGroup = require( 'SCENERY/nodes/AlignGroup' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Circle = require( 'SCENERY/nodes/Circle' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );

  QUnit.module( 'AlignBox' );

  QUnit.test( 'Single Box Group', function( assert ) {
    var circle = new Circle( 20 );

    var group = new AlignGroup();
    var box = group.createBox( circle, {
      xMargin: 10,
      yMargin: 20
    } );

    assert.ok( box.xMargin === 10, 'xMargin' );
    assert.ok( box.yMargin === 20, 'yMargin' );
    assert.ok( box.xAlign === 'center', 'xAlign' );
    assert.ok( box.yAlign === 'center', 'yAlign' );

    assert.ok( box.bounds.equals( new Bounds2( 0, 0, 60, 80 ) ), 'Margins' );
    assert.ok( circle.bounds.equals( new Bounds2( 10, 20, 50, 60 ) ), 'Circle: Margins' );

    circle.radius = 10;
    circle.getBounds(); // trigger bounds check
    assert.ok( box.bounds.equals( new Bounds2( 0, 0, 40, 60 ) ), 'Change to the content size' );
    assert.ok( circle.bounds.equals( new Bounds2( 10, 20, 30, 40 ) ), 'Circle: Change to the content size' );

    circle.x = 100;
    circle.getBounds(); // trigger bounds check
    assert.ok( box.bounds.equals( new Bounds2( 0, 0, 40, 60 ) ), 'Reposition on content location change' );
    assert.ok( circle.bounds.equals( new Bounds2( 10, 20, 30, 40 ) ), 'Circle: Reposition on content location change' );

    circle.scale( 2 );
    circle.getBounds(); // trigger bounds check
    assert.ok( box.bounds.equals( new Bounds2( 0, 0, 60, 80 ) ), 'Handle scaling' );
    assert.ok( circle.bounds.equals( new Bounds2( 10, 20, 50, 60 ) ), 'Circle: Handle scaling' );

    box.xMargin = 0;
    circle.getBounds(); // trigger bounds check
    assert.ok( box.bounds.equals( new Bounds2( 0, 0, 40, 80 ) ), 'xMargin change' );
    assert.ok( circle.bounds.equals( new Bounds2( 0, 20, 40, 60 ) ), 'Circle: xMargin change' );

    group.dispose();
  } );

  QUnit.test( 'Multiple Boxes in a Group', function( assert ) {
    var circle = new Circle( 10 );
    var rectangle = new Rectangle( 0, 0, 60, 60 );

    var group = new AlignGroup();

    var circleBox = group.createBox( circle, {
      xMargin: 10,
      yMargin: 20,
      xAlign: 'left',
      yAlign: 'bottom'
    } );

    var rectangleBox = group.createBox( rectangle );

    assert.ok( circleBox.xMargin === 10, 'circle: xMargin' );
    assert.ok( circleBox.yMargin === 20, 'circle: yMargin' );
    assert.ok( circleBox.xAlign === 'left', 'circle: xAlign' );
    assert.ok( circleBox.yAlign === 'bottom', 'circle: yAlign' );

    assert.ok( rectangleBox.xMargin === 0, 'rectangle: xMargin' );
    assert.ok( rectangleBox.yMargin === 0, 'rectangle: yMargin' );
    assert.ok( rectangleBox.xAlign === 'center', 'rectangle: xAlign' );
    assert.ok( rectangleBox.yAlign === 'center', 'rectangle: yAlign' );

    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60 (max in both)
    assert.ok( circleBox.bounds.equals( new Bounds2( 0, 0, 60, 60 ) ), 'Circle Container: Initial multiple' );
    assert.ok( rectangleBox.bounds.equals( new Bounds2( 0, 0, 60, 60 ) ), 'Rectangle Container: Initial multiple' );
    assert.ok( circle.bounds.equals( new Bounds2( 10, 20, 30, 40 ) ), 'Circle: Initial multiple' );
    assert.ok( rectangle.bounds.equals( new Bounds2( 0, 0, 60, 60 ) ), 'Rectangle: Initial multiple' );

    rectangleBox.yMargin = 20;
    circleBox.getBounds();
    rectangleBox.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    assert.ok( circleBox.bounds.equals( new Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: Align Change Rect' );
    assert.ok( rectangleBox.bounds.equals( new Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: Align Change Rect' );
    assert.ok( circle.bounds.equals( new Bounds2( 10, 60, 30, 80 ) ), 'Circle: Align Change Rect' );
    assert.ok( rectangle.bounds.equals( new Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: Align Change Rect' );

    circleBox.yAlign = 'top';
    circleBox.getBounds();
    rectangleBox.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    assert.ok( circleBox.bounds.equals( new Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: Align Change Circle' );
    assert.ok( rectangleBox.bounds.equals( new Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: Align Change Circle' );
    assert.ok( circle.bounds.equals( new Bounds2( 10, 20, 30, 40 ) ), 'Circle: Align Change Circle' );
    assert.ok( rectangle.bounds.equals( new Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: Align Change Circle' );

    rectangleBox.dispose();
    circleBox.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    assert.ok( circleBox.bounds.equals( new Bounds2( 0, 0, 40, 60 ) ), 'Circle Container: Removed Rect Container' );
    assert.ok( circle.bounds.equals( new Bounds2( 10, 20, 30, 40 ) ), 'Circle: Removed Rect Container' );

    rectangleBox = group.createBox( rectangle, { yMargin: 20 } );
    circleBox.getBounds();
    rectangleBox.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    assert.ok( circleBox.bounds.equals( new Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: Added back box' );
    assert.ok( rectangleBox.bounds.equals( new Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: Added back box' );
    assert.ok( circle.bounds.equals( new Bounds2( 10, 20, 30, 40 ) ), 'Circle: Added back box' );
    assert.ok( rectangle.bounds.equals( new Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: Added back box' );

    circleBox.xAlign = 'right';
    circleBox.getBounds();
    rectangleBox.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    assert.ok( circleBox.bounds.equals( new Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: More circle xAlign:right' );
    assert.ok( rectangleBox.bounds.equals( new Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: More circle xAlign:right' );
    assert.ok( circle.bounds.equals( new Bounds2( 30, 20, 50, 40 ) ), 'Circle: More circle xAlign:right' );
    assert.ok( rectangle.bounds.equals( new Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: More circle xAlign:right' );


    group.dispose();
  } );
} );