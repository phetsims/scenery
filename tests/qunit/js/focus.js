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

    var trails = root.getSerializedFocusOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    equal( trails.length, 4 );
  } );

  test( 'Visibility Test', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true, visible: false } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ], visible: false } );

    var root = new scenery.Node( { children: [ a, b ] } );

    var trails = root.getSerializedFocusOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    equal( trails.length, 1 );
  } );

  test( 'focusOrder Simple Test', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], focusOrder: [ b, a ] } );

    var trails = root.getSerializedFocusOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    equal( trails.length, 4 );
  } );

  test( 'focusOrder Descendant Test', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], focusOrder: [ a1, b1, a2, b2 ] } );

    var trails = root.getSerializedFocusOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    equal( trails.length, 4 );
  } );

  test( 'focusOrder Descendant Pruning Test', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var c1 = new scenery.Node( { focusable: true } );
    var c2 = new scenery.Node( { focusable: true } );

    var c = new scenery.Node( { children: [ c1, c2 ] } );

    var a = new scenery.Node( { children: [ a1, a2, c ] } );
    var b = new scenery.Node( { children: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], focusOrder: [ c1, a, a2, b2 ] } );

    var trails = root.getSerializedFocusOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, a, c, c1 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, a, c, c2 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    ok( trails[4].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    equal( trails.length, 5 );
  } );

  test( 'focusOrder Descendant Override', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ], focusOrder: [ b1, b2 ] } );

    var root = new scenery.Node( { children: [ a, b ], focusOrder: [ b, b1, a ] } );

    var trails = root.getSerializedFocusOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    equal( trails.length, 4 );
  } );

  test( 'focusOrder Hierarchy', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var b1 = new scenery.Node( { focusable: true } );
    var b2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ], focusOrder: [ a2 ] } );
    var b = new scenery.Node( { children: [ b1, b2 ], focusOrder: [ b2, b1 ] } );

    var root = new scenery.Node( { children: [ a, b ], focusOrder: [ b, a ] } );

    var trails = root.getSerializedFocusOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, b, b2 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, b, b1 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    equal( trails.length, 3 );
  } );

  test( 'focusOrder DAG test', function() {

    var a1 = new scenery.Node( { focusable: true } );
    var a2 = new scenery.Node( { focusable: true } );

    var a = new scenery.Node( { children: [ a1, a2 ], focusOrder: [ a2, a1 ] } );
    var b = new scenery.Node( { children: [ a1, a2 ], focusOrder: [ a1, a2 ] } );

    var root = new scenery.Node( { children: [ a, b ] } );

    var trails = root.getSerializedFocusOrder();

    ok( trails[0].equals( new scenery.Trail( [ root, a, a2 ] ) ) );
    ok( trails[1].equals( new scenery.Trail( [ root, a, a1 ] ) ) );
    ok( trails[2].equals( new scenery.Trail( [ root, b, a1 ] ) ) );
    ok( trails[3].equals( new scenery.Trail( [ root, b, a2 ] ) ) );
    equal( trails.length, 4 );
  } );
})();
