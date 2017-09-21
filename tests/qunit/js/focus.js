// Copyright 2016, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: Focus' );

  // Arrays of items of the type { trail: {Trail}, children: {Array.<Item>} }
  function nestedEquality( a, b ) {
    equal( a.length, b.length );

    for ( var i = 0; i < a.length; i++ ) {
      var aItem = a[ i ];
      var bItem = b[ i ];

      ok( aItem.trail.equals( bItem.trail ) );

      nestedEquality( aItem.children, bItem.children );
    }
  }

  var accessibleContent = {
    createPeer: function( accessibleInstance ) {
      return new scenery.AccessiblePeer( accessibleInstance, document.createElement( 'div' ) );
    }
  };

  test( 'Simple Test', function() {

    var a1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var a2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var b1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var b2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( nestedOrder, [
      { trail: new scenery.Trail( [ root, a, a1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a2 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, b1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, b2 ] ), children: [] }
    ] );
  } );

  test( 'accessibleOrder Simple Test', function() {

    var a1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var a2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var b1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var b2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], accessibleOrder: [ b, a ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( nestedOrder, [
      { trail: new scenery.Trail( [ root, b, b1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, b2 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a2 ] ), children: [] }
    ] );
  } );

  test( 'accessibleOrder Descendant Test', function() {

    var a1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var a2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var b1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var b2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], accessibleOrder: [ a1, b1, a2, b2 ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( nestedOrder, [
      { trail: new scenery.Trail( [ root, a, a1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, b1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a2 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, b2 ] ), children: [] }
    ] );
  } );

  test( 'accessibleOrder Descendant Pruning Test', function() {

    var a1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var a2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var b1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var b2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var c1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var c2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var c = new scenery.Node( { children: [ c1, c2 ] } );

    var a = new scenery.Node( { children: [ a1, a2, c ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], accessibleOrder: [ c1, a, a2, b2 ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( nestedOrder, [
      { trail: new scenery.Trail( [ root, a, c, c1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, c, c2 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a2 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, b2 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, b1 ] ), children: [] }
    ] );
  } );

  test( 'accessibleOrder Descendant Override', function() {

    var a1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var a2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var b1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var b2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ], accessibleOrder: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], accessibleOrder: [ b, b1, a ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( nestedOrder, [
      { trail: new scenery.Trail( [ root, b, b2 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, b1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a2 ] ), children: [] }
    ] );
  } );

  test( 'accessibleOrder Hierarchy', function() {

    var a1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var a2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var b1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var b2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var a = new scenery.Node( { children: [ a1, a2 ], accessibleOrder: [ a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ], accessibleOrder: [ b2, b1 ] } );

    var root = new scenery.Node( { children: [ a, b ], accessibleOrder: [ b, a ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( nestedOrder, [
      { trail: new scenery.Trail( [ root, b, b2 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, b1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a2 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a1 ] ), children: [] }
    ] );
  } );

  test( 'accessibleOrder DAG test', function() {

    var a1 = new scenery.Node( { accessibleContent: accessibleContent } );
    var a2 = new scenery.Node( { accessibleContent: accessibleContent } );

    var a = new scenery.Node( { children: [ a1, a2 ], accessibleOrder: [ a2, a1 ] } );
    var b = new scenery.Node( { children: [ a1, a2 ], accessibleOrder: [ a1, a2 ] } );

    var root = new scenery.Node( { children: [ a, b ] } );

    var nestedOrder = root.getNestedAccessibleOrder();

    nestedEquality( nestedOrder, [
      { trail: new scenery.Trail( [ root, a, a2 ] ), children: [] },
      { trail: new scenery.Trail( [ root, a, a1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, a1 ] ), children: [] },
      { trail: new scenery.Trail( [ root, b, a2 ] ), children: [] }
    ] );
  } );

  test( 'accessibleOrder DAG test', function() {

    var x = new scenery.Node();
    var a = new scenery.Node();
    var b = new scenery.Node();
    var c = new scenery.Node();
    var d = new scenery.Node( { accessibleContent: accessibleContent } );
    var e = new scenery.Node();
    var f = new scenery.Node( { accessibleContent: accessibleContent } );
    var g = new scenery.Node( { accessibleContent: accessibleContent } );
    var h = new scenery.Node( { accessibleContent: accessibleContent } );
    var i = new scenery.Node( { accessibleContent: accessibleContent } );
    var j = new scenery.Node( { accessibleContent: accessibleContent } );
    var k = new scenery.Node( { accessibleContent: accessibleContent } );
    var l = new scenery.Node();

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

    nestedEquality( nestedOrder, [
      // x order's F
      { trail: new scenery.Trail( [ x, a, b, e, f ] ), children: [
        { trail: new scenery.Trail( [ x, a, b, e, f, h ] ), children: [] },
        { trail: new scenery.Trail( [ x, a, b, e, f, i ] ), children: [] },
      ] },
      { trail: new scenery.Trail( [ x, a, c, e, f ] ), children: [
        { trail: new scenery.Trail( [ x, a, c, e, f, h ] ), children: [] },
        { trail: new scenery.Trail( [ x, a, c, e, f, i ] ), children: [] },
      ] },

      // X order's C
      { trail: new scenery.Trail( [ x, a, c, e, g ] ), children: [] },
      { trail: new scenery.Trail( [ x, a, c, e, j ] ), children: [] },

      // X order's D
      { trail: new scenery.Trail( [ x, a, b, d ] ), children: [] },

      // X everything else
      { trail: new scenery.Trail( [ x, a, b, e, g ] ), children: [] },
      { trail: new scenery.Trail( [ x, a, b, e, j ] ), children: [] },
      { trail: new scenery.Trail( [ x, a, k ] ), children: [] }
    ] );
  } );

  test( 'setting accessibleOrder', function() {

    var rootNode = new scenery.Node();
    var display = new scenery.Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );
    
    var a = new scenery.Node( { tagName: 'div' } );
    var b = new scenery.Node( { tagName: 'div' } );
    var c = new scenery.Node( { tagName: 'div' } );
    var d = new scenery.Node( { tagName: 'div' } );
    rootNode.children = [ a, b, c, d ];

    // reverse accessible order
    rootNode.accessibleOrder = [ d, c, b, a ];

    // debugger;
    var divRoot = display._rootAccessibleInstance.peer.domElement;
    var divA = a.accessibleInstances[ 0 ].peer.domElement;
    var divB = b.accessibleInstances[ 0 ].peer.domElement;
    var divC = c.accessibleInstances[ 0 ].peer.domElement;
    var divD = d.accessibleInstances[ 0 ].peer.domElement;

    ok( divRoot.children[ 0 ] === divD, 'divD should be first child' );
    ok( divRoot.children[ 1 ] === divC, 'divC should be second child' );
    ok( divRoot.children[ 2 ] === divB, 'divB should be third child' );
    ok( divRoot.children[ 3 ] === divA, 'divA should be fourth child' );
  } );

  test( 'setting accessibleOrder before setting accessible content', function() {
    var rootNode = new scenery.Node();
    var display = new scenery.Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );
    
    var a = new scenery.Node();
    var b = new scenery.Node();
    var c = new scenery.Node();
    var d = new scenery.Node();
    rootNode.children = [ a, b, c, d ];

    // reverse accessible order
    rootNode.accessibleOrder = [ d, c, b, a ];

    a.tagName = 'div';
    b.tagName = 'div';
    c.tagName = 'div';
    d.tagName = 'div';

    var divRoot = display._rootAccessibleInstance.peer.domElement;
    var divA = a.accessibleInstances[ 0 ].peer.domElement;
    var divB = b.accessibleInstances[ 0 ].peer.domElement;
    var divC = c.accessibleInstances[ 0 ].peer.domElement;
    var divD = d.accessibleInstances[ 0 ].peer.domElement;

    ok( divRoot.children[ 0 ] === divD, 'divD should be first child' );
    ok( divRoot.children[ 1 ] === divC, 'divC should be second child' );
    ok( divRoot.children[ 2 ] === divB, 'divB should be third child' );
    ok( divRoot.children[ 3 ] === divA, 'divA should be fourth child' );
  } );

  test( 'setting accessible order on nodes with no accessible content', function() {
    var rootNode = new scenery.Node();
    var display = new scenery.Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    // root
    //    a
    //      b
    //     c   e 
    //        d  f
    
    var a = new scenery.Node( { tagName: 'div' } );
    var b = new scenery.Node( { tagName: 'div' } );
    var c = new scenery.Node( { tagName: 'div' } );
    var d = new scenery.Node( { tagName: 'div' } );
    var e = new scenery.Node( { tagName: 'div' } );
    var f = new scenery.Node( { tagName: 'div' } );
    rootNode.addChild( a );
    a.addChild( b );
    b.addChild( c );
    b.addChild( e );
    c.addChild( d ); 
    c.addChild( f );

    var divB = b.accessibleInstances[ 0 ].peer.domElement;
    var divC = c.accessibleInstances[ 0 ].peer.domElement;
    var divE = e.accessibleInstances[ 0 ].peer.domElement;

    b.accessibleOrder = [ e, c ];

    ok( divB.children[ 0 ] === divE, 'div E should be first child of div B' );
    ok( divB.children[ 1 ] === divC, 'div C should be second child of div B' );
  } );

  test( 'setting accessible order on nodes with no accessible content', function() {
    var rootNode = new scenery.Node();
    var display = new scenery.Display( rootNode ); // eslint-disable-line
    document.body.appendChild( display.domElement );

    var a = new scenery.Node( { tagName: 'div' } );
    var b = new scenery.Node();
    var c = new scenery.Node( { tagName: 'div' } );
    var d = new scenery.Node( { tagName: 'div' } );
    var e = new scenery.Node( { tagName: 'div' } );
    var f = new scenery.Node( { tagName: 'div' } );
    rootNode.addChild( a );
    a.addChild( b );
    b.addChild( c );
    b.addChild( e );
    c.addChild( d ); 
    c.addChild( f );

    var divA = a.accessibleInstances[ 0 ].peer.domElement;
    var divC = c.accessibleInstances[ 0 ].peer.domElement;
    var divE = e.accessibleInstances[ 0 ].peer.domElement;

    a.accessibleOrder = [ e, c ];
    ok( divA.children[ 0 ] === divE, 'div E should be first child of div B' );
    ok( divA.children[ 1 ] === divC, 'div C should be second child of div B' );
  } );
})();
