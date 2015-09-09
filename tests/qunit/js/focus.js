(function() {
  'use strict';

  module( 'Scenery: Focus' );

  test( 'Simple Test', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ] } );

    var trails = root.getSerializedAccessibleOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    equal( trails.length, 4 );
  } );

  test( 'accessibleOrder Simple Test', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], accessibleOrder: [ b, a ] } );

    var trails = root.getSerializedAccessibleOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    equal( trails.length, 4 );
  } );

  test( 'accessibleOrder Descendant Test', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], accessibleOrder: [ a1, b1, a2, b2 ] } );

    var trails = root.getSerializedAccessibleOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    equal( trails.length, 4 );
  } );

  test( 'accessibleOrder Descendant Pruning Test', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var c1 = new scenery.Node( { focusable: true } );
    var c2 = new scenery.Node( { focusable: true } );

    var c = new scenery.Node( { children: [ c1, c2 ] } );

    var a = new scenery.Node( { children: [ a1, a2, c ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], accessibleOrder: [ c1, a, a2, b2 ] } );

    var trails = root.getSerializedAccessibleOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, a, c, c1 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, a, c, c2 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    ok( trails[4].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    ok( trails[5].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    equal( trails.length, 6 );
  } );

  test( 'accessibleOrder Descendant Override', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ], accessibleOrder: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], accessibleOrder: [ b, b1, a ] } );

    var trails = root.getSerializedAccessibleOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    equal( trails.length, 4 );
  } );

  test( 'accessibleOrder Hierarchy', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ], accessibleOrder: [ a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ], accessibleOrder: [ b2, b1 ] } );

    var root = new scenery.Node( { children: [ a, b ], accessibleOrder: [ b, a ] } );

    var trails = root.getSerializedAccessibleOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, a, a1 ] ) ) );

    equal( trails.length, 4 );
  } );

  test( 'accessibleOrder DAG test', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ], accessibleOrder: [ a2, a1 ] } );
    var b = new scenery.Node( { children: [ a1, a2 ], accessibleOrder: [ a1, a2 ] } );

    var root = new scenery.Node( { children: [ a, b ] } );

    var trails = root.getSerializedAccessibleOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, b, a1 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, b, a2 ] ) ) );
    equal( trails.length, 4 );
  } );

  test( 'accessibleOrder DAG test', function() {

    var x = new scenery.Node();
    var a = new scenery.Node();
    var b = new scenery.Node();
    var c = new scenery.Node();
    var d = new scenery.Node( { focusable: true } );
    var e = new scenery.Node();
    var f = new scenery.Node( { focusable: true } );
    var g = new scenery.Node( { focusable: true } );
    var h = new scenery.Node( { focusable: true } );
    var i = new scenery.Node( { focusable: true } );
    var j = new scenery.Node( { focusable: true } );
    var k = new scenery.Node( { focusable: true } );
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

    var trails = x.getSerializedAccessibleOrder();

    // x order's F
    ok( trails[0].equals( new scenery.Trail( [ x, a, b, e, f ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ x, a, b, e, f, h ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ x, a, b, e, f, i ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ x, a, c, e, f ] ) ) );
    ok( trails[4].equals( new scenery.Trail( [ x, a, c, e, f, h ] ) ) );
    ok( trails[5].equals( new scenery.Trail( [ x, a, c, e, f, i ] ) ) );

    // X order's C
    ok( trails[6].equals( new scenery.Trail( [ x, a, c, e, g ] ) ) );
    ok( trails[7].equals( new scenery.Trail( [ x, a, c, e, j ] ) ) );

    // X order's D
    ok( trails[8].equals( new scenery.Trail( [ x, a, b, d ] ) ) );

    // X everything else
    ok( trails[9].equals( new scenery.Trail( [ x, a, b, e, g ] ) ) );
    ok( trails[10].equals( new scenery.Trail( [ x, a, b, e, j ] ) ) );
    ok( trails[11].equals( new scenery.Trail( [ x, a, k ] ) ) );

    equal( trails.length, 12 );
  } );
})();
