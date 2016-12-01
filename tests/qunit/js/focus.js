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
})();
