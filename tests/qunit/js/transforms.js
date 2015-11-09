// Copyright 2002-2014, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: Transforms' );

  var epsilon = 0.000000001;

  test( 'Points (parent and child)', function() {
    var a = new scenery.Node();
    var b = new scenery.Node();
    a.addChild( b );
    a.x = 10;
    b.y = 10;

    ok( dot.v2( 5, 15 ).equalsEpsilon( b.localToParentPoint( dot.v2( 5, 5 ) ), epsilon ), 'localToParentPoint on child' );
    ok( dot.v2( 15, 5 ).equalsEpsilon( a.localToParentPoint( dot.v2( 5, 5 ) ), epsilon ), 'localToParentPoint on root' );

    ok( dot.v2( 5, -5 ).equalsEpsilon( b.parentToLocalPoint( dot.v2( 5, 5 ) ), epsilon ), 'parentToLocalPoint on child' );
    ok( dot.v2( -5, 5 ).equalsEpsilon( a.parentToLocalPoint( dot.v2( 5, 5 ) ), epsilon ), 'parentToLocalPoint on root' );

    ok( dot.v2( 15, 15 ).equalsEpsilon( b.localToGlobalPoint( dot.v2( 5, 5 ) ), epsilon ), 'localToGlobalPoint on child' );
    ok( dot.v2( 15, 5 ).equalsEpsilon( a.localToGlobalPoint( dot.v2( 5, 5 ) ), epsilon ), 'localToGlobalPoint on root (same as localToparent)' );

    ok( dot.v2( -5, -5 ).equalsEpsilon( b.globalToLocalPoint( dot.v2( 5, 5 ) ), epsilon ), 'globalToLocalPoint on child' );
    ok( dot.v2( -5, 5 ).equalsEpsilon( a.globalToLocalPoint( dot.v2( 5, 5 ) ), epsilon ), 'globalToLocalPoint on root (same as localToparent)' );

    ok( dot.v2( 15, 5 ).equalsEpsilon( b.parentToGlobalPoint( dot.v2( 5, 5 ) ), epsilon ), 'parentToGlobalPoint on child' );
    ok( dot.v2( 5, 5 ).equalsEpsilon( a.parentToGlobalPoint( dot.v2( 5, 5 ) ), epsilon ), 'parentToGlobalPoint on root' );

    ok( dot.v2( -5, 5 ).equalsEpsilon( b.globalToParentPoint( dot.v2( 5, 5 ) ), epsilon ), 'globalToParentPoint on child' );
    ok( dot.v2( 5, 5 ).equalsEpsilon( a.globalToParentPoint( dot.v2( 5, 5 ) ), epsilon ), 'globalToParentPoint on root' );

  } );

  test( 'Bounds (parent and child)', function() {
    var a = new scenery.Node();
    var b = new scenery.Node();
    a.addChild( b );
    a.x = 10;
    b.y = 10;

    var Bounds2 = dot.Bounds2;

    var bounds = new Bounds2( 4, 4, 20, 30 );

    ok( new Bounds2( 4, 14, 20, 40 ).equalsEpsilon( b.localToParentBounds( bounds ), epsilon ), 'localToParentBounds on child' );
    ok( new Bounds2( 14, 4, 30, 30 ).equalsEpsilon( a.localToParentBounds( bounds ), epsilon ), 'localToParentBounds on root' );

    ok( new Bounds2( 4, -6, 20, 20 ).equalsEpsilon( b.parentToLocalBounds( bounds ), epsilon ), 'parentToLocalBounds on child' );
    ok( new Bounds2( -6, 4, 10, 30 ).equalsEpsilon( a.parentToLocalBounds( bounds ), epsilon ), 'parentToLocalBounds on root' );

    ok( new Bounds2( 14, 14, 30, 40 ).equalsEpsilon( b.localToGlobalBounds( bounds ), epsilon ), 'localToGlobalBounds on child' );
    ok( new Bounds2( 14, 4, 30, 30 ).equalsEpsilon( a.localToGlobalBounds( bounds ), epsilon ), 'localToGlobalBounds on root (same as localToParent)' );

    ok( new Bounds2( -6, -6, 10, 20 ).equalsEpsilon( b.globalToLocalBounds( bounds ), epsilon ), 'globalToLocalBounds on child' );
    ok( new Bounds2( -6, 4, 10, 30 ).equalsEpsilon( a.globalToLocalBounds( bounds ), epsilon ), 'globalToLocalBounds on root (same as localToParent)' );

    ok( new Bounds2( 14, 4, 30, 30 ).equalsEpsilon( b.parentToGlobalBounds( bounds ), epsilon ), 'parentToGlobalBounds on child' );
    ok( new Bounds2( 4, 4, 20, 30 ).equalsEpsilon( a.parentToGlobalBounds( bounds ), epsilon ), 'parentToGlobalBounds on root' );

    ok( new Bounds2( -6, 4, 10, 30 ).equalsEpsilon( b.globalToParentBounds( bounds ), epsilon ), 'globalToParentBounds on child' );
    ok( new Bounds2( 4, 4, 20, 30 ).equalsEpsilon( a.globalToParentBounds( bounds ), epsilon ), 'globalToParentBounds on root' );
  } );

  test( 'Points (order of transforms)', function() {
    var a = new scenery.Node();
    var b = new scenery.Node();
    var c = new scenery.Node();
    a.addChild( b );
    b.addChild( c );
    a.x = 10;
    b.scale( 2 );
    c.y = 10;

    ok( dot.v2( 20, 30 ).equalsEpsilon( c.localToGlobalPoint( dot.v2( 5, 5 ) ), epsilon ), 'localToGlobalPoint' );
    ok( dot.v2( -2.5, -7.5 ).equalsEpsilon( c.globalToLocalPoint( dot.v2( 5, 5 ) ), epsilon ), 'globalToLocalPoint' );
    ok( dot.v2( 20, 10 ).equalsEpsilon( c.parentToGlobalPoint( dot.v2( 5, 5 ) ), epsilon ), 'parentToGlobalPoint' );
    ok( dot.v2( -2.5, 2.5 ).equalsEpsilon( c.globalToParentPoint( dot.v2( 5, 5 ) ), epsilon ), 'globalToParentPoint' );
  } );

  test( 'Bounds (order of transforms)', function() {
    var a = new scenery.Node();
    var b = new scenery.Node();
    var c = new scenery.Node();
    a.addChild( b );
    b.addChild( c );
    a.x = 10;
    b.scale( 2 );
    c.y = 10;

    var Bounds2 = dot.Bounds2;

    var bounds = new Bounds2( 4, 4, 20, 30 );

    ok( new Bounds2( 18, 28, 50, 80 ).equalsEpsilon( c.localToGlobalBounds( bounds ), epsilon ), 'localToGlobalBounds' );
    ok( new Bounds2( -3, -8, 5, 5 ).equalsEpsilon( c.globalToLocalBounds( bounds ), epsilon ), 'globalToLocalBounds' );
    ok( new Bounds2( 18, 8, 50, 60 ).equalsEpsilon( c.parentToGlobalBounds( bounds ), epsilon ), 'parentToGlobalBounds' );
    ok( new Bounds2( -3, 2, 5, 15 ).equalsEpsilon( c.globalToParentBounds( bounds ), epsilon ), 'globalToParentBounds' );
  } );

  test( 'Trail and Node transform equivalence', function() {
    var a = new scenery.Node();
    var b = new scenery.Node();
    var c = new scenery.Node();
    a.addChild( b );
    b.addChild( c );
    a.x = 10;
    b.scale( 2 );
    c.y = 10;

    var trailMatrix = c.getUniqueTrail().getMatrix();
    var nodeMatrix = c.getUniqueTransform().getMatrix();
    ok( trailMatrix.equalsEpsilon( nodeMatrix, epsilon ), 'Trail and Node transform equivalence' );
  } );

})();
