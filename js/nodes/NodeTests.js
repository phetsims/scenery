// Copyright 2017, University of Colorado Boulder

/**
 * Node tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var Bounds2 = require( 'DOT/Bounds2' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Shape = require( 'KITE/Shape' );
  var Touch = require( 'SCENERY/input/Touch' );
  var Vector2 = require( 'DOT/Vector2' );

  QUnit.module( 'Node' );

  function fakeTouchPointer( vector ) {
    return new Touch( 0, vector, {} );
  }

  QUnit.test( 'Mouse and Touch areas', function( assert ) {
    var node = new Node();
    var rect = new Rectangle( 0, 0, 100, 50 );
    rect.pickable = true;

    node.addChild( rect );

    assert.ok( !!rect.hitTest( new Vector2( 10, 10 ) ), 'Rectangle intersection' );
    assert.ok( !!rect.hitTest( new Vector2( 90, 10 ) ), 'Rectangle intersection' );
    assert.ok( !rect.hitTest( new Vector2( -10, 10 ) ), 'Rectangle no intersection' );

    node.touchArea = Shape.rectangle( -50, -50, 100, 100 );

    assert.ok( !!node.hitTest( new Vector2( 10, 10 ) ), 'Node intersection' );
    assert.ok( !!node.hitTest( new Vector2( 90, 10 ) ), 'Node intersection' );
    assert.ok( !node.hitTest( new Vector2( -10, 10 ) ), 'Node no intersection' );

    assert.ok( !!node.trailUnderPointer( fakeTouchPointer( new Vector2( 10, 10 ) ) ), 'Node intersection (isTouch)' );
    assert.ok( !!node.trailUnderPointer( fakeTouchPointer( new Vector2( 90, 10 ) ) ), 'Node intersection (isTouch)' );
    assert.ok( !!node.trailUnderPointer( fakeTouchPointer( new Vector2( -10, 10 ) ) ), 'Node intersection (isTouch)' );

    node.clipArea = Shape.rectangle( 0, 0, 50, 50 );

    // points outside the clip area shouldn't register as hits
    assert.ok( !!node.trailUnderPointer( fakeTouchPointer( new Vector2( 10, 10 ) ) ), 'Node intersection (isTouch with clipArea)' );
    assert.ok( !node.trailUnderPointer( fakeTouchPointer( new Vector2( 90, 10 ) ) ), 'Node no intersection (isTouch with clipArea)' );
    assert.ok( !node.trailUnderPointer( fakeTouchPointer( new Vector2( -10, 10 ) ) ), 'Node no intersection (isTouch with clipArea)' );
  } );


  var epsilon = 0.000000001;

  QUnit.test( 'Points (parent and child)', function( assert ) {
    var a = new Node();
    var b = new Node();
    a.addChild( b );
    a.x = 10;
    b.y = 10;

    assert.ok( new Vector2( 5, 15 ).equalsEpsilon( b.localToParentPoint( new Vector2( 5, 5 ) ), epsilon ), 'localToParentPoint on child' );
    assert.ok( new Vector2( 15, 5 ).equalsEpsilon( a.localToParentPoint( new Vector2( 5, 5 ) ), epsilon ), 'localToParentPoint on root' );

    assert.ok( new Vector2( 5, -5 ).equalsEpsilon( b.parentToLocalPoint( new Vector2( 5, 5 ) ), epsilon ), 'parentToLocalPoint on child' );
    assert.ok( new Vector2( -5, 5 ).equalsEpsilon( a.parentToLocalPoint( new Vector2( 5, 5 ) ), epsilon ), 'parentToLocalPoint on root' );

    assert.ok( new Vector2( 15, 15 ).equalsEpsilon( b.localToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'localToGlobalPoint on child' );
    assert.ok( new Vector2( 15, 5 ).equalsEpsilon( a.localToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'localToGlobalPoint on root (same as localToparent)' );

    assert.ok( new Vector2( -5, -5 ).equalsEpsilon( b.globalToLocalPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToLocalPoint on child' );
    assert.ok( new Vector2( -5, 5 ).equalsEpsilon( a.globalToLocalPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToLocalPoint on root (same as localToparent)' );

    assert.ok( new Vector2( 15, 5 ).equalsEpsilon( b.parentToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'parentToGlobalPoint on child' );
    assert.ok( new Vector2( 5, 5 ).equalsEpsilon( a.parentToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'parentToGlobalPoint on root' );

    assert.ok( new Vector2( -5, 5 ).equalsEpsilon( b.globalToParentPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToParentPoint on child' );
    assert.ok( new Vector2( 5, 5 ).equalsEpsilon( a.globalToParentPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToParentPoint on root' );

  } );

  QUnit.test( 'Bounds (parent and child)', function( assert ) {
    var a = new Node();
    var b = new Node();
    a.addChild( b );
    a.x = 10;
    b.y = 10;

    var bounds = new Bounds2( 4, 4, 20, 30 );

    assert.ok( new Bounds2( 4, 14, 20, 40 ).equalsEpsilon( b.localToParentBounds( bounds ), epsilon ), 'localToParentBounds on child' );
    assert.ok( new Bounds2( 14, 4, 30, 30 ).equalsEpsilon( a.localToParentBounds( bounds ), epsilon ), 'localToParentBounds on root' );

    assert.ok( new Bounds2( 4, -6, 20, 20 ).equalsEpsilon( b.parentToLocalBounds( bounds ), epsilon ), 'parentToLocalBounds on child' );
    assert.ok( new Bounds2( -6, 4, 10, 30 ).equalsEpsilon( a.parentToLocalBounds( bounds ), epsilon ), 'parentToLocalBounds on root' );

    assert.ok( new Bounds2( 14, 14, 30, 40 ).equalsEpsilon( b.localToGlobalBounds( bounds ), epsilon ), 'localToGlobalBounds on child' );
    assert.ok( new Bounds2( 14, 4, 30, 30 ).equalsEpsilon( a.localToGlobalBounds( bounds ), epsilon ), 'localToGlobalBounds on root (same as localToParent)' );

    assert.ok( new Bounds2( -6, -6, 10, 20 ).equalsEpsilon( b.globalToLocalBounds( bounds ), epsilon ), 'globalToLocalBounds on child' );
    assert.ok( new Bounds2( -6, 4, 10, 30 ).equalsEpsilon( a.globalToLocalBounds( bounds ), epsilon ), 'globalToLocalBounds on root (same as localToParent)' );

    assert.ok( new Bounds2( 14, 4, 30, 30 ).equalsEpsilon( b.parentToGlobalBounds( bounds ), epsilon ), 'parentToGlobalBounds on child' );
    assert.ok( new Bounds2( 4, 4, 20, 30 ).equalsEpsilon( a.parentToGlobalBounds( bounds ), epsilon ), 'parentToGlobalBounds on root' );

    assert.ok( new Bounds2( -6, 4, 10, 30 ).equalsEpsilon( b.globalToParentBounds( bounds ), epsilon ), 'globalToParentBounds on child' );
    assert.ok( new Bounds2( 4, 4, 20, 30 ).equalsEpsilon( a.globalToParentBounds( bounds ), epsilon ), 'globalToParentBounds on root' );
  } );

  QUnit.test( 'Points (order of transforms)', function( assert ) {
    var a = new Node();
    var b = new Node();
    var c = new Node();
    a.addChild( b );
    b.addChild( c );
    a.x = 10;
    b.scale( 2 );
    c.y = 10;

    assert.ok( new Vector2( 20, 30 ).equalsEpsilon( c.localToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'localToGlobalPoint' );
    assert.ok( new Vector2( -2.5, -7.5 ).equalsEpsilon( c.globalToLocalPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToLocalPoint' );
    assert.ok( new Vector2( 20, 10 ).equalsEpsilon( c.parentToGlobalPoint( new Vector2( 5, 5 ) ), epsilon ), 'parentToGlobalPoint' );
    assert.ok( new Vector2( -2.5, 2.5 ).equalsEpsilon( c.globalToParentPoint( new Vector2( 5, 5 ) ), epsilon ), 'globalToParentPoint' );
  } );

  QUnit.test( 'Bounds (order of transforms)', function( assert ) {
    var a = new Node();
    var b = new Node();
    var c = new Node();
    a.addChild( b );
    b.addChild( c );
    a.x = 10;
    b.scale( 2 );
    c.y = 10;

    var bounds = new Bounds2( 4, 4, 20, 30 );

    assert.ok( new Bounds2( 18, 28, 50, 80 ).equalsEpsilon( c.localToGlobalBounds( bounds ), epsilon ), 'localToGlobalBounds' );
    assert.ok( new Bounds2( -3, -8, 5, 5 ).equalsEpsilon( c.globalToLocalBounds( bounds ), epsilon ), 'globalToLocalBounds' );
    assert.ok( new Bounds2( 18, 8, 50, 60 ).equalsEpsilon( c.parentToGlobalBounds( bounds ), epsilon ), 'parentToGlobalBounds' );
    assert.ok( new Bounds2( -3, 2, 5, 15 ).equalsEpsilon( c.globalToParentBounds( bounds ), epsilon ), 'globalToParentBounds' );
  } );

  QUnit.test( 'Trail and Node transform equivalence', function( assert ) {
    var a = new Node();
    var b = new Node();
    var c = new Node();
    a.addChild( b );
    b.addChild( c );
    a.x = 10;
    b.scale( 2 );
    c.y = 10;

    var trailMatrix = c.getUniqueTrail().getMatrix();
    var nodeMatrix = c.getUniqueTransform().getMatrix();
    assert.ok( trailMatrix.equalsEpsilon( nodeMatrix, epsilon ), 'Trail and Node transform equivalence' );
  } );
} );