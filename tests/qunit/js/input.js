// Copyright 2002-2014, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: Input' );

  test( 'Mouse and Touch areas', function() {
    var node = new scenery.Node();
    var rect = new scenery.Rectangle( 0, 0, 100, 50 );
    rect.pickable = true;

    node.addChild( rect );

    ok( rect.trailUnderPoint( dot.v2( 10, 10 ) ), 'Rectangle intersection' );
    ok( rect.trailUnderPoint( dot.v2( 90, 10 ) ), 'Rectangle intersection' );
    ok( !rect.trailUnderPoint( dot.v2( -10, 10 ) ), 'Rectangle no intersection' );

    node.touchArea = kite.Shape.rectangle( -50, -50, 100, 100 );

    ok( node.trailUnderPoint( dot.v2( 10, 10 ) ), 'Node intersection' );
    ok( node.trailUnderPoint( dot.v2( 90, 10 ) ), 'Node intersection' );
    ok( !node.trailUnderPoint( dot.v2( -10, 10 ) ), 'Node no intersection' );

    ok( node.trailUnderPointer( { isTouch: true, point: dot.v2( 10, 10 ) } ), 'Node intersection (isTouch)' );
    ok( node.trailUnderPointer( { isTouch: true, point: dot.v2( 90, 10 ) } ), 'Node intersection (isTouch)' );
    ok( node.trailUnderPointer( { isTouch: true, point: dot.v2( -10, 10 ) } ), 'Node intersection (isTouch)' );

    node.clipArea = kite.Shape.rectangle( 0, 0, 50, 50 );

    // points outside the clip area shouldn't register as hits
    ok( node.trailUnderPointer( { isTouch: true, point: dot.v2( 10, 10 ) } ), 'Node intersection (isTouch with clipArea)' );
    ok( !node.trailUnderPointer( { isTouch: true, point: dot.v2( 90, 10 ) } ), 'Node no intersection (isTouch with clipArea)' );
    ok( !node.trailUnderPointer( { isTouch: true, point: dot.v2( -10, 10 ) } ), 'Node no intersection (isTouch with clipArea)' );
  } );
})();
