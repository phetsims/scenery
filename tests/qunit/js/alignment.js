// Copyright 2002-2016, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: AlignBox/AlignGroup' );

  test( 'Single Container', function() {
    var circle = new scenery.Circle( 20 );

    var group = new scenery.AlignGroup();
    var container = group.createBox( circle, {
      xMargin: 10,
      yMargin: 20
    } );

    ok( container.xMargin === 10, 'xMargin' );
    ok( container.yMargin === 20, 'yMargin' );
    ok( container.xAlign === 'center', 'xAlign' );
    ok( container.yAlign === 'center', 'yAlign' );

    ok( container.bounds.equals( new dot.Bounds2( 0, 0, 60, 80 ) ), 'Margins' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 50, 60 ) ), 'Circle: Margins' );

    circle.radius = 10; circle.getBounds(); // trigger bounds check
    ok( container.bounds.equals( new dot.Bounds2( 0, 0, 40, 60 ) ), 'Change to the content size' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Change to the content size' );

    circle.x = 100; circle.getBounds(); // trigger bounds check
    ok( container.bounds.equals( new dot.Bounds2( 0, 0, 40, 60 ) ), 'Reposition on content location change' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Reposition on content location change' );

    circle.scale( 2 ); circle.getBounds(); // trigger bounds check
    ok( container.bounds.equals( new dot.Bounds2( 0, 0, 60, 80 ) ), 'Handle scaling' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 50, 60 ) ), 'Circle: Handle scaling' );

    container.xMargin = 0; circle.getBounds(); // trigger bounds check
    ok( container.bounds.equals( new dot.Bounds2( 0, 0, 40, 80 ) ), 'xMargin change' );
    ok( circle.bounds.equals( new dot.Bounds2( 0, 20, 40, 60 ) ), 'Circle: xMargin change' );

    group.dispose();
  } );

  test( 'Multiple Containers With Alignment', function() {
    var circle = new scenery.Circle( 10 );
    var rectangle = new scenery.Rectangle( 0, 0, 60, 60 );

    var group = new scenery.AlignGroup();

    var circleContainer = group.createBox( circle, {
      xMargin: 10,
      yMargin: 20,
      xAlign: 'left',
      yAlign: 'bottom'
    } );

    var rectangleContainer = group.createBox( rectangle );

    ok( circleContainer.xMargin === 10, 'circle: xMargin' );
    ok( circleContainer.yMargin === 20, 'circle: yMargin' );
    ok( circleContainer.xAlign === 'left', 'circle: xAlign' );
    ok( circleContainer.yAlign === 'bottom', 'circle: yAlign' );

    ok( rectangleContainer.xMargin === 0, 'rectangle: xMargin' );
    ok( rectangleContainer.yMargin === 0, 'rectangle: yMargin' );
    ok( rectangleContainer.xAlign === 'center', 'rectangle: xAlign' );
    ok( rectangleContainer.yAlign === 'center', 'rectangle: yAlign' );

    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60 (max in both)
    ok( circleContainer.bounds.equals( new dot.Bounds2( 0, 0, 60, 60 ) ), 'Circle Container: Initial multiple' );
    ok( rectangleContainer.bounds.equals( new dot.Bounds2( 0, 0, 60, 60 ) ), 'Rectangle Container: Initial multiple' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Initial multiple' );
    ok( rectangle.bounds.equals( new dot.Bounds2( 0, 0, 60, 60 ) ), 'Rectangle: Initial multiple' );

    rectangleContainer.yMargin = 20; circleContainer.getBounds(); rectangleContainer.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    ok( circleContainer.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: Align Change Rect' );
    ok( rectangleContainer.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: Align Change Rect' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 60, 30, 80 ) ), 'Circle: Align Change Rect' );
    ok( rectangle.bounds.equals( new dot.Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: Align Change Rect' );

    circleContainer.yAlign = 'top'; circleContainer.getBounds(); rectangleContainer.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    ok( circleContainer.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: Align Change Circle' );
    ok( rectangleContainer.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: Align Change Circle' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Align Change Circle' );
    ok( rectangle.bounds.equals( new dot.Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: Align Change Circle' );

    rectangleContainer.dispose(); circleContainer.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    ok( circleContainer.bounds.equals( new dot.Bounds2( 0, 0, 40, 60 ) ), 'Circle Container: Removed Rect Container' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Removed Rect Container' );

    rectangleContainer = group.createBox( rectangle, { yMargin: 20 } ); circleContainer.getBounds(); rectangleContainer.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    ok( circleContainer.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: Added back container' );
    ok( rectangleContainer.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: Added back container' );
    ok( circle.bounds.equals( new dot.Bounds2( 10, 20, 30, 40 ) ), 'Circle: Added back container' );
    ok( rectangle.bounds.equals( new dot.Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: Added back container' );

    circleContainer.xAlign = 'right'; circleContainer.getBounds(); rectangleContainer.getBounds(); // trigger check
    // circle: 20x20, with margin: 40x60
    // rectangle: 60x60, with margin: 60x100
    ok( circleContainer.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'Circle Container: More circle xAlign:right' );
    ok( rectangleContainer.bounds.equals( new dot.Bounds2( 0, 0, 60, 100 ) ), 'RectangleContainer: More circle xAlign:right' );
    ok( circle.bounds.equals( new dot.Bounds2( 30, 20, 50, 40 ) ), 'Circle: More circle xAlign:right' );
    ok( rectangle.bounds.equals( new dot.Bounds2( 0, 20, 60, 80 ) ), 'Rectangle: More circle xAlign:right' );


    group.dispose();
  } );
})();
