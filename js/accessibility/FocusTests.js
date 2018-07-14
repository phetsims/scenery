// Copyright 2017, University of Colorado Boulder

/**
 * Focus tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var Display = require( 'SCENERY/display/Display' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Trail = require( 'SCENERY/util/Trail' );

  QUnit.module( 'Focus' );

// Arrays of items of the type { trail: {Trail}, children: {Array.<Item>} }
  function nestedEquality( assert, a, b ) {
    assert.equal( a.length, b.length );

    for ( var i = 0; i < a.length; i++ ) {
      var aItem = a[ i ];
      var bItem = b[ i ];

      assert.ok( aItem.trail.equals( bItem.trail ) );

      nestedEquality( assert, aItem.children, bItem.children );
    }
  }

  QUnit.test( 'Simple Test', function( assert ) {

    var a1 = new Node( { tagName: 'div' } );
    var a2 = new Node( { tagName: 'div' } );

    var b1 = new Node( { tagName: 'div' } );
    var b2 = new Node( { tagName: 'div' } );

    var a = new Node( { children: [ a1, a2 ] } );
    var b = new Node( { children: [ b1, b2 ] } );

    var root = new Node( { children: [ a, b ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( assert, nestedOrder, [
      { trail: new Trail( [ root, a, a1 ] ), children: [] },
      { trail: new Trail( [ root, a, a2 ] ), children: [] },
      { trail: new Trail( [ root, b, b1 ] ), children: [] },
      { trail: new Trail( [ root, b, b2 ] ), children: [] }
    ] );
  } );

  QUnit.test( 'accessibleOrder Simple Test', function( assert ) {

    var a1 = new Node( { tagName: 'div' } );
    var a2 = new Node( { tagName: 'div' } );

    var b1 = new Node( { tagName: 'div' } );
    var b2 = new Node( { tagName: 'div' } );

    var a = new Node( { children: [ a1, a2 ] } );
    var b = new Node( { children: [ b1, b2 ] } );

    var root = new Node( { children: [ a, b ], accessibleOrder: [ b, a ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( assert, nestedOrder, [
      { trail: new Trail( [ root, b, b1 ] ), children: [] },
      { trail: new Trail( [ root, b, b2 ] ), children: [] },
      { trail: new Trail( [ root, a, a1 ] ), children: [] },
      { trail: new Trail( [ root, a, a2 ] ), children: [] }
    ] );
  } );

  QUnit.test( 'accessibleOrder Descendant Test', function( assert ) {

    var a1 = new Node( { tagName: 'div' } );
    var a2 = new Node( { tagName: 'div' } );

    var b1 = new Node( { tagName: 'div' } );
    var b2 = new Node( { tagName: 'div' } );

    var a = new Node( { children: [ a1, a2 ] } );
    var b = new Node( { children: [ b1, b2 ] } );

    var root = new Node( { children: [ a, b ], accessibleOrder: [ a1, b1, a2, b2 ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( assert, nestedOrder, [
      { trail: new Trail( [ root, a, a1 ] ), children: [] },
      { trail: new Trail( [ root, b, b1 ] ), children: [] },
      { trail: new Trail( [ root, a, a2 ] ), children: [] },
      { trail: new Trail( [ root, b, b2 ] ), children: [] }
    ] );
  } );

  QUnit.test( 'accessibleOrder Descendant Pruning Test', function( assert ) {

    var a1 = new Node( { tagName: 'div' } );
    var a2 = new Node( { tagName: 'div' } );

    var b1 = new Node( { tagName: 'div' } );
    var b2 = new Node( { tagName: 'div' } );

    var c1 = new Node( { tagName: 'div' } );
    var c2 = new Node( { tagName: 'div' } );

    var c = new Node( { children: [ c1, c2 ] } );

    var a = new Node( { children: [ a1, a2, c ] } );
    var b = new Node( { children: [ b1, b2 ] } );

    var root = new Node( { children: [ a, b ], accessibleOrder: [ c1, a, a2, b2 ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( assert, nestedOrder, [
      { trail: new Trail( [ root, a, c, c1 ] ), children: [] },
      { trail: new Trail( [ root, a, a1 ] ), children: [] },
      { trail: new Trail( [ root, a, c, c2 ] ), children: [] },
      { trail: new Trail( [ root, a, a2 ] ), children: [] },
      { trail: new Trail( [ root, b, b2 ] ), children: [] },
      { trail: new Trail( [ root, b, b1 ] ), children: [] }
    ] );
  } );

  QUnit.test( 'accessibleOrder Descendant Override', function( assert ) {

    var a1 = new Node( { tagName: 'div' } );
    var a2 = new Node( { tagName: 'div' } );

    var b1 = new Node( { tagName: 'div' } );
    var b2 = new Node( { tagName: 'div' } );

    var a = new Node( { children: [ a1, a2 ] } );
    var b = new Node( { children: [ b1, b2 ], accessibleOrder: [ b1, b2 ] } );

    var root = new Node( { children: [ a, b ], accessibleOrder: [ b, b1, a ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( assert, nestedOrder, [
      { trail: new Trail( [ root, b, b2 ] ), children: [] },
      { trail: new Trail( [ root, b, b1 ] ), children: [] },
      { trail: new Trail( [ root, a, a1 ] ), children: [] },
      { trail: new Trail( [ root, a, a2 ] ), children: [] }
    ] );
  } );

  QUnit.test( 'accessibleOrder Hierarchy', function( assert ) {

    var a1 = new Node( { tagName: 'div' } );
    var a2 = new Node( { tagName: 'div' } );

    var b1 = new Node( { tagName: 'div' } );
    var b2 = new Node( { tagName: 'div' } );

    var a = new Node( { children: [ a1, a2 ], accessibleOrder: [ a2 ] } );
    var b = new Node( { children: [ b1, b2 ], accessibleOrder: [ b2, b1 ] } );

    var root = new Node( { children: [ a, b ], accessibleOrder: [ b, a ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( assert, nestedOrder, [
      { trail: new Trail( [ root, b, b2 ] ), children: [] },
      { trail: new Trail( [ root, b, b1 ] ), children: [] },
      { trail: new Trail( [ root, a, a2 ] ), children: [] },
      { trail: new Trail( [ root, a, a1 ] ), children: [] }
    ] );
  } );

  QUnit.test( 'accessibleOrder DAG test', function( assert ) {

    var a1 = new Node( { tagName: 'div' } );
    var a2 = new Node( { tagName: 'div' } );

    var a = new Node( { children: [ a1, a2 ], accessibleOrder: [ a2, a1 ] } );
    var b = new Node( { children: [ a1, a2 ], accessibleOrder: [ a1, a2 ] } );

    var root = new Node( { children: [ a, b ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( assert, nestedOrder, [
      { trail: new Trail( [ root, a, a2 ] ), children: [] },
      { trail: new Trail( [ root, a, a1 ] ), children: [] },
      { trail: new Trail( [ root, b, a1 ] ), children: [] },
      { trail: new Trail( [ root, b, a2 ] ), children: [] }
    ] );
  } );

  QUnit.test( 'accessibleOrder DAG test', function( assert ) {

    var x = new Node();
    var a = new Node();
    var b = new Node();
    var c = new Node();
    var d = new Node( { tagName: 'div' } );
    var e = new Node();
    var f = new Node( { tagName: 'div' } );
    var g = new Node( { tagName: 'div' } );
    var h = new Node( { tagName: 'div' } );
    var i = new Node( { tagName: 'div' } );
    var j = new Node( { tagName: 'div' } );
    var k = new Node( { tagName: 'div' } );
    var l = new Node();

    x.children = [ a ];
    a.children = [ k, b, c ];
    b.children = [ d, e ];
    c.children = [ e ];
    e.children = [ j, f, g ];
    f.children = [ h, i ];

    x.accessibleOrder = [ f, c, d, l ];
    a.accessibleOrder = [ c, b ];
    e.accessibleOrder = [ g, f, j ];

    var nestedOrder = x.getNestedAccessibleOrder();

    nestedEquality( assert, nestedOrder, [
      // x order's F
      {
        trail: new Trail( [ x, a, b, e, f ] ), children: [
          { trail: new Trail( [ x, a, b, e, f, h ] ), children: [] },
          { trail: new Trail( [ x, a, b, e, f, i ] ), children: [] }
        ]
      },
      {
        trail: new Trail( [ x, a, c, e, f ] ), children: [
          { trail: new Trail( [ x, a, c, e, f, h ] ), children: [] },
          { trail: new Trail( [ x, a, c, e, f, i ] ), children: [] }
        ]
      },

      // X order's C
      { trail: new Trail( [ x, a, c, e, g ] ), children: [] },
      { trail: new Trail( [ x, a, c, e, j ] ), children: [] },

      // X order's D
      { trail: new Trail( [ x, a, b, d ] ), children: [] },

      // X everything else
      { trail: new Trail( [ x, a, b, e, g ] ), children: [] },
      { trail: new Trail( [ x, a, b, e, j ] ), children: [] },
      { trail: new Trail( [ x, a, k ] ), children: [] }
    ] );
  } );

  QUnit.test( 'setting accessibleOrder', function( assert ) {

    var rootNode = new Node();
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new Node( { tagName: 'div' } );
    var b = new Node( { tagName: 'div' } );
    var c = new Node( { tagName: 'div' } );
    var d = new Node( { tagName: 'div' } );
    rootNode.children = [ a, b, c, d ];

    // reverse accessible order
    rootNode.accessibleOrder = [ d, c, b, a ];

    var divRoot = display._rootAccessibleInstance.peer.primarySibling;
    var divA = a.accessibleInstances[ 0 ].peer.primarySibling;
    var divB = b.accessibleInstances[ 0 ].peer.primarySibling;
    var divC = c.accessibleInstances[ 0 ].peer.primarySibling;
    var divD = d.accessibleInstances[ 0 ].peer.primarySibling;

    assert.ok( divRoot.children[ 0 ] === divD, 'divD should be first child' );
    assert.ok( divRoot.children[ 1 ] === divC, 'divC should be second child' );
    assert.ok( divRoot.children[ 2 ] === divB, 'divB should be third child' );
    assert.ok( divRoot.children[ 3 ] === divA, 'divA should be fourth child' );
  } );

  QUnit.test( 'setting accessibleOrder before setting accessible content', function( assert ) {
    var rootNode = new Node();
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new Node();
    var b = new Node();
    var c = new Node();
    var d = new Node();
    rootNode.children = [ a, b, c, d ];

    // reverse accessible order
    rootNode.accessibleOrder = [ d, c, b, a ];

    a.tagName = 'div';
    b.tagName = 'div';
    c.tagName = 'div';
    d.tagName = 'div';

    var divRoot = display._rootAccessibleInstance.peer.primarySibling;
    var divA = a.accessibleInstances[ 0 ].peer.primarySibling;
    var divB = b.accessibleInstances[ 0 ].peer.primarySibling;
    var divC = c.accessibleInstances[ 0 ].peer.primarySibling;
    var divD = d.accessibleInstances[ 0 ].peer.primarySibling;

    assert.ok( divRoot.children[ 0 ] === divD, 'divD should be first child' );
    assert.ok( divRoot.children[ 1 ] === divC, 'divC should be second child' );
    assert.ok( divRoot.children[ 2 ] === divB, 'divB should be third child' );
    assert.ok( divRoot.children[ 3 ] === divA, 'divA should be fourth child' );
  } );

  QUnit.test( 'setting accessible order on nodes with no accessible content', function( assert ) {
    var rootNode = new Node();
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // root
    //    a
    //      b
    //     c   e
    //        d  f

    var a = new Node( { tagName: 'div' } );
    var b = new Node( { tagName: 'div' } );
    var c = new Node( { tagName: 'div' } );
    var d = new Node( { tagName: 'div' } );
    var e = new Node( { tagName: 'div' } );
    var f = new Node( { tagName: 'div' } );
    rootNode.addChild( a );
    a.addChild( b );
    b.addChild( c );
    b.addChild( e );
    c.addChild( d );
    c.addChild( f );
    b.accessibleOrder = [ e, c ];

    var divB = b.accessibleInstances[ 0 ].peer.primarySibling;
    var divC = c.accessibleInstances[ 0 ].peer.primarySibling;
    var divE = e.accessibleInstances[ 0 ].peer.primarySibling;

    assert.ok( divB.children[ 0 ] === divE, 'div E should be first child of div B' );
    assert.ok( divB.children[ 1 ] === divC, 'div C should be second child of div B' );
  } );

  QUnit.test( 'setting accessible order on nodes with no accessible content', function( assert ) {
    var rootNode = new Node();
    var display = new Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new Node( { tagName: 'div' } );
    var b = new Node();
    var c = new Node( { tagName: 'div' } );
    var d = new Node( { tagName: 'div' } );
    var e = new Node( { tagName: 'div' } );
    var f = new Node( { tagName: 'div' } );
    rootNode.addChild( a );
    a.addChild( b );
    b.addChild( c );
    b.addChild( e );
    c.addChild( d );
    c.addChild( f );
    a.accessibleOrder = [ e, c ];

    var divA = a.accessibleInstances[ 0 ].peer.primarySibling;
    var divC = c.accessibleInstances[ 0 ].peer.primarySibling;
    var divE = e.accessibleInstances[ 0 ].peer.primarySibling;

    assert.ok( divA.children[ 0 ] === divE, 'div E should be first child of div B' );
    assert.ok( divA.children[ 1 ] === divC, 'div C should be second child of div B' );
  } );
} );