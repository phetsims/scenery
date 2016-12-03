// Copyright 2002-2016, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: AlignBox/AlignGroup' );

  test( 'Single Box Group', function() {
    var circle = new scenery.Circle( 20 );

    var group = new scenery.AlignGroup();
    var box = group.createBox( circle, {
      xMargin: 10,
      yMargin: 20
    } );

    ok( box.xMargin === 10, 'xMargin' );
    ok( box.yMargin === 20, 'yMargin' );
    ok( box.xAlign === 'center', 'xAlign' );
    ok( box.yAlign === 'center', 'yAlign' );

    ok( box.bounds.equals( new dot.Bounds2( 0, 0, 60, 80 ) ), 'Margins' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 50, 60 ) ), 'Circle: Margins' );

    circle.radius = 10; circle.getBounds(); // trigger bounds check
    ok( box.bounds.equals( new dot.Bounds2( 0, 0, 40, 60 ) ), 'Change to the content size' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Change to the content size' );

    circle.x = 100; circle.getBounds(); // trigger bounds check
    ok( box.bounds.equals( new dot.Bounds2( 0, 0, 40, 60 ) ), 'Reposition on content location change' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Reposition on content location change' );

    circle.scale( 2 ); circle.getBounds(); // trigger bounds check
    ok( box.bounds.equals( new dot.Bounds2( 0, 0, 60, 80 ) ), 'Handle scaling' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 50, 60 ) ), 'Circle: Handle scaling' );

    box.xMargin = 0; circle.getBounds(); // trigger bounds check
    ok( box.bounds.equals( new dot.Bounds2( 0, 0, 40, 80 ) ), 'xMargin change' );
    ok( circle.bounds.equals( new dot.Bounds2( 0, 20, 40, 60 ) ), 'Circle: xMargin change' );

    group.dispose();
  } );

  test( 'Multiple Boxes in a Group', function() {
    var circle = new scenery.Circle( 10 );
    var rectangle = new scenery.Rectangle( 0, 0, 60, 60 );

    var group = new scenery.AlignGroup();

    var circleBox = group.createBox( circle, {
      xMargin: 10,
      yMargin: 20,
      xAlign: 'left',
      yAlign: 'bottom'
    } );

    var rectangleBox = group.createBox( rectangle );

    ok( circleBox.xMargin === 10, 'circle: xMargin' );
    ok( circleBox.yMargin === 20, 'circle: yMargin' );
    ok( circleBox.xAlign === 'left', 'circle: xAlign' );
    ok( circleBox.yAlign === 'bottom', 'circle: yAlign' );

    ok( rectangleBox.xMargin === 0, 'rectangle: xMargin' );
    ok( rectangleBox.yMargin === 0, 'rectangle: yMargin' );
    ok( rectangleBox.xAlign === 'center', 'rectangle: xAlign' );
    ok( rectangleBox.yAlign === 'center', 'rectangle: yAlign' );

    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60 (max in both)
    ok( circleBox.bounds.equals( new dot.Bounds2( 0, 0, 60, 60 ) ), 'Circle Container: Initial multiple' );
    ok( rectangleBox.bounds.equals( new dot.Bounds2( 0, 0, 60, 60 ) ), 'Rectangle Container: Initial multiple' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Initial multiple' );
    ok( rectangle.bounds.equals( new dot.Bounds2( 0, 0, 60, 60 ) ), 'Rectangle: Initial multiple' );

    rectangleBox.yMargin = 20; circleBox.getBounds(); rectangleBox.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    ok( circleBox.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: Align Change Rect' );
    ok( rectangleBox.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: Align Change Rect' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 60, 30, 80 ) ), 'Circle: Align Change Rect' );
    ok( rectangle.bounds.equals( new dot.Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: Align Change Rect' );

    circleBox.yAlign = 'top'; circleBox.getBounds(); rectangleBox.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    ok( circleBox.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: Align Change Circle' );
    ok( rectangleBox.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: Align Change Circle' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Align Change Circle' );
    ok( rectangle.bounds.equals( new dot.Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: Align Change Circle' );

    rectangleBox.dispose(); circleBox.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    ok( circleBox.bounds.equals( new dot.Bounds2( 0, 0, 40, 60 ) ), 'Circle Container: Removed Rect Container' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Removed Rect Container' );

    rectangleBox = group.createBox( rectangle, { yMargin: 20 } ); circleBox.getBounds(); rectangleBox.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    ok( circleBox.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: Added back box' );
    ok( rectangleBox.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: Added back box' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Added back box' );
    ok( rectangle.bounds.equals( new dot.Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: Added back box' );

    circleBox.xAlign = 'right'; circleBox.getBounds(); rectangleBox.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    ok( circleBox.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: More circle xAlign:right' );
    ok( rectangleBox.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: More circle xAlign:right' );
    ok( circle.bounds.equals( new dot.Bounds2( 30, 20, 50, 40 ) ), 'Circle: More circle xAlign:right' );
    ok( rectangle.bounds.equals( new dot.Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: More circle xAlign:right' );


    group.dispose();
  } );
})();
