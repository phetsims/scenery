
(function(){
  'use strict';
  
  
  
  module( 'Scenery: Layering' );
  
  test( 'Layer quantity check', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    scene.layerAudit();
    
    equal( scene.layers.length, 0, 'no layers at the start' );
    
    var a = new scenery.Path();
    var b = new scenery.Path();
    var c = new scenery.Path();
    scene.addChild( a );
    scene.layerAudit();
    scene.addChild( b );
    scene.layerAudit();
    scene.addChild( c );
    scene.layerAudit();
    
    equal( scene.layers.length, 1, 'just a single layer for three paths' );
    ok( scene.layerLookup( a.getUniqueTrail() ) === scene.layers[0] );
    ok( scene.layerLookup( b.getUniqueTrail() ) === scene.layers[0] );
    ok( scene.layerLookup( c.getUniqueTrail() ) === scene.layers[0] );
    
    var d = new scenery.Path();
    b.addChild( d );
    
    scene.layerAudit();
    equal( scene.layers.length, 1, 'still just a single layer' );
    ok( scene.layerLookup( a.getUniqueTrail() ) === scene.layers[0] );
    ok( scene.layerLookup( b.getUniqueTrail() ) === scene.layers[0] );
    ok( scene.layerLookup( c.getUniqueTrail() ) === scene.layers[0] );
    
    b.renderer = 'canvas';
    
    scene.layerAudit();
    equal( scene.layers.length, 1, 'scene is canvas, so b should not trigger any more layers' );
    ok( scene.layerLookup( a.getUniqueTrail() ) === scene.layers[0] );
    ok( scene.layerLookup( b.getUniqueTrail() ) === scene.layers[0] );
    ok( scene.layerLookup( c.getUniqueTrail() ) === scene.layers[0] );
    
    b.renderer = 'svg';
    
    scene.layerAudit();
    equal( scene.layers.length, 3, 'should be canvas, svg, canvas' );
    ok( scene.layerLookup( a.getUniqueTrail() ) === scene.layers[0] );
    ok( scene.layerLookup( b.getUniqueTrail() ) === scene.layers[1] );
    ok( scene.layerLookup( c.getUniqueTrail() ) === scene.layers[2] );
    
    c.renderer = 'svg';
    
    scene.layerAudit();
    equal( scene.layers.length, 2, 'should be canvas, svg (combined)' );
    ok( scene.layerLookup( a.getUniqueTrail() ) === scene.layers[0] );
    ok( scene.layerLookup( b.getUniqueTrail() ) === scene.layers[1] );
    ok( scene.layerLookup( c.getUniqueTrail() ) === scene.layers[1] );
    
    b.rendererOptions = {
      someUniqueThingToThisLayer: 5
    };
    
    scene.layerAudit();
    equal( scene.layers.length, 3, 'should be canvas, svg (with options), svg' );
    ok( scene.layerLookup( a.getUniqueTrail() ) === scene.layers[0] );
    ok( scene.layerLookup( b.getUniqueTrail() ) === scene.layers[1] );
    ok( scene.layerLookup( c.getUniqueTrail() ) === scene.layers[2] );
  } );
  
  // TODO: occurs with layer matching
  test( 'endBoundary failing case', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    var n4 = new scenery.Node();
    var n5 = new scenery.Node();
    var n7 = new scenery.Node();
    var p9 = new scenery.Path();
    var p10 = new scenery.Path();
    scene.addChild( p9 );
    scene.layerAudit();
    scene.addChild( n7 );
    scene.layerAudit();
    scene.addChild( n4 );
    scene.layerAudit();
    p9.addChild( p10 );
    scene.layerAudit();
    p9.addChild( n4 );
    scene.layerAudit();
    n7.addChild( n4 );
    scene.layerAudit();
    n4.addChild( n5 );
    scene.layerAudit();
    n4.addChild( p10 );
    scene.layerAudit();
    n4.removeChild( p10 );
    scene.layerAudit();
    
    expect( 0 );
  } );
  
  test( 'unknown break #1', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    var p1 = new scenery.Path();
    var p2 = new scenery.Path();
    var p3 = new scenery.Path();
    var n1 = new scenery.Node();
    
    scene.addChild( p1 );
    scene.addChild( p2 );
    p1.addChild( n1 );
    p1.addChild( p3 );
    n1.addChild( p3 );
    p2.renderer = 'svg';
    
    expect( 0 );
  } );
  
  test( 'unknown break #2', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    var node3 = new scenery.Node( {} );
    var node4 = new scenery.Node( {} );
    var node5 = new scenery.Node( {} );
    var node6 = new scenery.Node( {} );
    var node7 = new scenery.Node( {} );
    var path8 = new scenery.Path( {} );
    var path9 = new scenery.Path( {} );
    var path10 = new scenery.Path( {} );
    var path11 = new scenery.Path( {} );
    var path12 = new scenery.Path( {} );
    var path13 = new scenery.Path( {} );
    var path14 = new scenery.Path( {} );
    var path15 = new scenery.Path( {} );
    var path16 = new scenery.Path( {} );
    var path17 = new scenery.Path( {} );
    function op( f ) {
      f();
      scene.layerAudit();
    }
    
    // reproducible test case from fuzzing
    op( function() { path9.insertChild( 0, path14 ); } );
    op( function() { path12.renderer = 'svg'; } );
    op( function() { node6.insertChild( 0, path14 ); } );
    op( function() { path11.insertChild( 0, node6 ); } );
    op( function() { node3.insertChild( 0, node6 ); } );
    op( function() { path15.renderer = null; } );
    op( function() { node4.insertChild( 0, node5 ); } );
    op( function() { node6.removeChild( path14 ); } );
    op( function() { path9.removeChild( path14 ); } );
    op( function() { path11.insertChild( 0, path15 ); } );
    op( function() { path10.renderer = 'canvas'; } );
    op( function() { path14.insertChild( 0, node6 ); } );
    op( function() { path16.insertChild( 0, node7 ); } );
    op( function() { path14.insertChild( 0, node7 ); } );
    op( function() { path14.removeChild( node6 ); } );
    op( function() { node3.renderer = null; } );
    op( function() { path13.renderer = 'canvas'; } );
    op( function() { node5.renderer = null; } );
    op( function() { path16.renderer = null; } );
    op( function() { path10.renderer = 'canvas'; } );
    op( function() { scene.renderer = null; } );
    op( function() { path10.renderer = null; } );
    op( function() { path16.renderer = 'canvas'; } );
    op( function() { path10.renderer = null; } );
    op( function() { path10.renderer = 'canvas'; } );
    op( function() { path11.removeChild( path15 ); } );
    op( function() { node4.removeChild( node5 ); } );
    op( function() { path10.insertChild( 0, path11 ); } );
    op( function() { path17.insertChild( 0, path8 ); } );
    op( function() { path9.renderer = null; } );
    op( function() { path9.renderer = null; } );
    op( function() { scene.insertChild( 0, node4 ); } );
    op( function() { path17.removeChild( path8 ); } );
    op( function() { path14.removeChild( node7 ); } );
    op( function() { path14.insertChild( 0, node5 ); } );
    op( function() { path14.renderer = 'canvas'; } );
    op( function() { path16.insertChild( 1, path11 ); } );
    op( function() { node4.renderer = 'canvas'; } );
    op( function() { path12.insertChild( 0, node6 ); } );
    op( function() { path11.removeChild( node6 ); } );
    op( function() { path14.renderer = null; } );
    op( function() { path13.renderer = null; } );
    op( function() { node7.renderer = 'svg'; } );
    op( function() { node4.renderer = 'canvas'; } );
    op( function() { path13.renderer = 'svg'; } );
    op( function() { path10.renderer = 'canvas'; } );
    op( function() { path8.renderer = 'svg'; } );
    op( function() { path12.removeChild( node6 ); } );
    op( function() { node5.renderer = null; } );
    op( function() { path15.renderer = 'canvas'; } );
    op( function() { path13.renderer = null; } );
    op( function() { path16.removeChild( node7 ); } );
    op( function() { path17.insertChild( 0, path16 ); } );
    op( function() { path14.insertChild( 0, node7 ); } );
    op( function() { node3.removeChild( node6 ); } );
    op( function() { scene.renderer = 'svg'; } );
    op( function() { path11.insertChild( 0, node3 ); } );
    op( function() { path16.renderer = 'canvas'; } );
    op( function() { path12.renderer = 'svg'; } );
    op( function() { path10.insertChild( 0, path16 ); } );
    op( function() { node5.insertChild( 0, path10 ); } );
    op( function() { path10.renderer = null; } );
    op( function() { scene.renderer = null; } );
    op( function() { node3.renderer = null; } );
    op( function() { path15.renderer = 'canvas'; } );
    op( function() { node3.renderer = 'canvas'; } );
    op( function() { node4.renderer = 'svg'; } );
    op( function() { path11.removeChild( node3 ); } );
    op( function() { node4.renderer = 'svg'; } );
    op( function() { path17.renderer = 'canvas'; } );
    op( function() { path15.renderer = null; } );
    op( function() { node6.insertChild( 0, path11 ); } );
    op( function() { path9.insertChild( 0, node4 ); } );
    op( function() { path8.insertChild( 0, path9 ); } );
    op( function() { node5.renderer = null; } );
    op( function() { path8.insertChild( 1, path16 ); } );
    op( function() { scene.removeChild( node4 ); } );
    op( function() { path15.insertChild( 0, path11 ); } );
    op( function() { node4.insertChild( 0, node6 ); } );
    op( function() { path13.renderer = null; } );
    op( function() { node6.insertChild( 0, path17 ); } );
    op( function() { path16.removeChild( path11 ); } );
    op( function() { scene.insertChild( 0, path10 ); } );
    op( function() { path15.insertChild( 0, path13 ); } );
    op( function() { path10.removeChild( path11 ); } );
    op( function() { scene.removeChild( path10 ); } );
    op( function() { path14.insertChild( 2, path16 ); } );
    op( function() { node4.renderer = 'canvas'; } );
    op( function() { path12.renderer = 'svg'; } );
    op( function() { path10.renderer = 'svg'; } );
    op( function() { node6.insertChild( 0, scene ); } );
    op( function() { path16.renderer = null; } );
    op( function() { scene.renderer = null; } );
    op( function() { path10.insertChild( 0, node3 ); } );
    op( function() { path13.insertChild( 0, path16 ); } );
    op( function() { path8.renderer = 'canvas'; } );
    op( function() { path17.renderer = 'canvas'; } );
    op( function() { node4.removeChild( node6 ); } );
    op( function() { path17.removeChild( path16 ); } );
    op( function() { node5.removeChild( path10 ); } );
    op( function() { path8.insertChild( 0, path13 ); } );
    op( function() { node7.renderer = null; } );
    op( function() { scene.renderer = null; } );
    op( function() { node6.removeChild( path17 ); } );
    op( function() { node7.insertChild( 0, node3 ); } );
    op( function() { path15.removeChild( path13 ); } );
    op( function() { path9.removeChild( node4 ); } );
    op( function() { path9.renderer = null; } );
    op( function() { path16.insertChild( 0, scene ); } );
    op( function() { node7.insertChild( 0, path9 ); } );
    op( function() { path15.removeChild( path11 ); } );
    op( function() { scene.renderer = 'svg'; } );
    op( function() { path16.renderer = 'canvas'; } );
    op( function() { path12.insertChild( 0, path8 ); } );
    op( function() { path15.renderer = 'svg'; } );
    op( function() { path13.renderer = null; } );
    op( function() { path17.renderer = 'svg'; } );
    op( function() { path17.insertChild( 0, path9 ); } );
    op( function() { node5.renderer = 'svg'; } );
    op( function() { path9.insertChild( 0, path13 ); } );
    op( function() { path17.insertChild( 0, path11 ); } );
    op( function() { path17.removeChild( path9 ); } );
    op( function() { path14.insertChild( 0, path8 ); } );
    op( function() { path15.insertChild( 0, path16 ); } );
    op( function() { path15.insertChild( 0, path13 ); } );
    op( function() { path12.renderer = null; } );
    op( function() { node4.renderer = null; } );
    op( function() { path15.removeChild( path13 ); } );
    op( function() { node4.insertChild( 0, node7 ); } );
    op( function() { path11.renderer = 'canvas'; } );
    op( function() { node6.renderer = 'canvas'; } );
    op( function() { node4.removeChild( node7 ); } );
    op( function() { path9.removeChild( path13 ); } );
    op( function() { scene.renderer = 'svg'; } );
    op( function() { path8.renderer = null; } );
    op( function() { path16.removeChild( scene ); } );
    op( function() { path15.removeChild( path16 ); } );
    op( function() { path8.renderer = 'canvas'; } );
    op( function() { scene.renderer = 'svg'; } );
    op( function() { node6.renderer = 'svg'; } );
    op( function() { path13.renderer = null; } );
    op( function() { path8.renderer = null; } );
    op( function() { node3.insertChild( 0, path12 ); } );
    op( function() { path8.removeChild( path9 ); } );
    op( function() { path10.insertChild( 1, path12 ); } );
    op( function() { scene.renderer = 'canvas'; } );
    op( function() { path17.insertChild( 0, node4 ); } );
    op( function() { path12.removeChild( path8 ); } );
    op( function() { path14.renderer = null; } );
    op( function() { node6.renderer = null; } );
    op( function() { path14.removeChild( path8 ); } );
    op( function() { node3.renderer = 'canvas'; } );
    op( function() { node7.insertChild( 2, path12 ); } );
    op( function() { node4.renderer = 'canvas'; } );
    op( function() { path11.insertChild( 0, node5 ); } );
    op( function() { path11.renderer = null; } );
    op( function() { path10.removeChild( path12 ); } );
    op( function() { node3.renderer = 'canvas'; } );
    op( function() { path11.removeChild( node5 ); } );
    op( function() { path13.removeChild( path16 ); } );
    op( function() { path8.renderer = null; } );
    op( function() { path12.insertChild( 0, node6 ); } );
    op( function() { path10.insertChild( 2, path8 ); } );
    op( function() { path8.insertChild( 2, path11 ); } );
    op( function() { node7.insertChild( 2, node6 ); } );
    op( function() { path16.renderer = 'svg'; } );
    op( function() { path17.renderer = null; } );
    op( function() { node7.renderer = 'canvas'; } );
    op( function() { path16.insertChild( 0, node5 ); } );
    op( function() { node7.removeChild( node3 ); } );
    op( function() { path17.removeChild( node4 ); } );
    op( function() { path10.insertChild( 2, node5 ); } );
    op( function() { node3.renderer = 'svg'; } );
    op( function() { node6.removeChild( path11 ); } );
    op( function() { path17.insertChild( 0, path14 ); } );
    op( function() { path14.insertChild( 0, path9 ); } );
    op( function() { path17.removeChild( path14 ); } );
    op( function() { path17.removeChild( path11 ); } );
    op( function() { path14.insertChild( 0, path17 ); } );
    op( function() { path9.renderer = null; } );
    op( function() { path11.insertChild( 0, path13 ); } );
    op( function() { path10.insertChild( 1, path14 ); } );
    op( function() { path14.removeChild( path9 ); } );
    op( function() { path10.renderer = null; } );
    op( function() { node7.insertChild( 2, path17 ); } );
    op( function() { node3.removeChild( path12 ); } );
    op( function() { node6.insertChild( 0, path11 ); } );
    op( function() { node6.renderer = null; } );
    op( function() { path13.insertChild( 0, scene ); } );
    op( function() { path9.renderer = null; } );
    op( function() { node7.renderer = null; } );
    op( function() { path13.removeChild( scene ); } );
    op( function() { node3.insertChild( 0, node7 ); } );
    op( function() { node3.removeChild( node7 ); } );
    op( function() { node4.insertChild( 0, path16 ); } );
    op( function() { node6.insertChild( 2, path9 ); } );
    op( function() { path14.removeChild( path16 ); } );
    op( function() { path8.removeChild( path13 ); } );
    op( function() { path14.removeChild( node7 ); } );
    op( function() { node7.insertChild( 1, path13 ); } );
    op( function() { path15.renderer = 'canvas'; } );
    op( function() { path14.insertChild( 2, path15 ); } );
    op( function() { node5.insertChild( 0, path17 ); } );
    op( function() { node4.insertChild( 1, scene ); } );
    op( function() { path12.renderer = null; } );
    op( function() { path14.removeChild( path17 ); } );
    op( function() { node5.removeChild( path17 ); } );
    op( function() { path13.renderer = 'svg'; } );
    op( function() { node3.renderer = null; } );
    op( function() { node5.renderer = null; } );
    op( function() { node4.renderer = null; } );
    op( function() { path17.insertChild( 0, node4 ); } );
    op( function() { path8.insertChild( 0, path9 ); } );
    op( function() { path17.renderer = 'svg'; } );
    op( function() { node4.renderer = null; } );
    op( function() { node5.renderer = null; } );
    op( function() { node5.renderer = null; } );
    op( function() { path14.removeChild( path15 ); } );
    op( function() { path9.renderer = 'canvas'; } );
    op( function() { scene.insertChild( 0, path15 ); } );
    op( function() { path8.removeChild( path11 ); } );
    op( function() { path14.insertChild( 1, path13 ); } );
    op( function() { path12.removeChild( node6 ); } );
    op( function() { path17.renderer = 'canvas'; } );
    op( function() { path9.renderer = null; } );
    op( function() { path16.insertChild( 1, path9 ); } );
    op( function() { path9.insertChild( 0, path15 ); } );
    op( function() { node6.renderer = 'svg'; } );
    op( function() { node3.renderer = 'canvas'; } );
    op( function() { node4.renderer = 'canvas'; } );
    op( function() { node4.insertChild( 1, path8 ); } );
    op( function() { path9.renderer = null; } );
    op( function() { node6.insertChild( 0, path16 ); } );
    op( function() { path17.renderer = null; } );
    op( function() { path10.renderer = null; } );
    op( function() { node4.insertChild( 3, node6 ); } );
    op( function() { path9.renderer = null; } );
    op( function() { path14.renderer = null; } );
    op( function() { path11.insertChild( 0, path9 ); } );
    op( function() { path9.removeChild( path15 ); } );
    op( function() { path11.renderer = null; } );
    op( function() { path9.insertChild( 0, path15 ); } );
    op( function() { path9.renderer = 'svg'; } );
    op( function() { path10.renderer = 'canvas'; } );
    op( function() { path15.renderer = null; } );
    op( function() { path12.insertChild( 0, path17 ); } );
    op( function() { path12.removeChild( path17 ); } );
    op( function() { path8.removeChild( path16 ); } );
    op( function() { node4.renderer = 'canvas'; } );
    op( function() { path16.removeChild( path9 ); } );
    op( function() { path14.renderer = 'svg'; } );
    op( function() { node5.insertChild( 0, path9 ); } );
    op( function() { path11.renderer = 'canvas'; } );
    op( function() { path11.renderer = null; } );
    op( function() { path14.renderer = null; } );
    op( function() { node7.removeChild( node6 ); } );
    op( function() { scene.removeChild( path15 ); } );
    op( function() { path8.removeChild( path9 ); } );
    op( function() { node7.renderer = null; } );
    op( function() { path16.renderer = null; } );
    op( function() { node4.renderer = 'svg'; } );
    op( function() { node6.removeChild( scene ); } );
    op( function() { node6.insertChild( 2, path8 ); } );
    op( function() { node6.removeChild( path16 ); } );
    op( function() { path8.insertChild( 0, path16 ); } );
    op( function() { node3.renderer = null; } );
    op( function() { node6.insertChild( 3, path13 ); } );
    op( function() { node4.insertChild( 1, node5 ); } );
    op( function() { path9.renderer = null; } );
    op( function() { path16.insertChild( 0, path11 ); } );
    op( function() { path17.renderer = 'svg'; } );
    op( function() { node5.removeChild( path9 ); } );
    op( function() { path13.insertChild( 0, path9 ); } );
    op( function() { path12.renderer = null; } );
    op( function() { path9.removeChild( path15 ); } );
    op( function() { path10.insertChild( 2, path15 ); } );
    op( function() { path14.insertChild( 2, path9 ); } );
    op( function() { node6.renderer = 'svg'; } );
    op( function() { node7.renderer = null; } );
    op( function() { node4.removeChild( path8 ); } );
    op( function() { node3.insertChild( 0, node5 ); } );
    op( function() { node6.renderer = null; } );
    op( function() { path15.insertChild( 0, node3 ); } );
    op( function() { path13.removeChild( path9 ); } );
    op( function() { path14.renderer = 'canvas'; } );
    op( function() { path15.removeChild( node3 ); } );
    op( function() { path8.removeChild( path16 ); } );
    op( function() { path13.renderer = null; } );
    op( function() { path14.renderer = 'canvas'; } );
    op( function() { path16.renderer = null; } );
    op( function() { path14.removeChild( path13 ); } );
    op( function() { path13.insertChild( 0, path14 ); } );
    op( function() { path14.removeChild( node5 ); } );
    op( function() { node7.renderer = 'canvas'; } );
    op( function() { scene.insertChild( 0, path10 ); } );
    op( function() { path9.renderer = null; } );
    op( function() { path8.renderer = 'svg'; } );
    op( function() { node4.insertChild( 0, node3 ); } );
    op( function() { node3.renderer = 'svg'; } );
    op( function() { scene.renderer = null; } );
    
    expect( 0 );
  } );
  
  test( 'Single subnode renderer toggle', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    var a = new scenery.Path();
    scene.addChild( a );
    scene.layerAudit();
    a.renderer = 'canvas'; // it was canvas before, and the layer seems to disappear!
    scene.layerAudit();
    
    equal( scene.layers.length, 1, 'should still have a layer' );
  } );
  
  test( 'Two-node inversion', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    scene.layerAudit();
    
    var path1 = new scenery.Path();
    var path2 = new scenery.Path();
    
    scene.addChild( path1 );
    scene.layerAudit();
    scene.addChild( path2 );
    scene.layerAudit();
    path1.renderer = 'svg';
    scene.layerAudit();
    
    expect( 0 );
  } );
  
  test( 'Three-node renderer toggles', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    scene.layerAudit();
    
    var path1 = new scenery.Path();
    var path2 = new scenery.Path();
    var path3 = new scenery.Path();
    
    scene.addChild( path1 );
    scene.layerAudit();
    
    scene.addChild( path2 );
    scene.layerAudit();
    
    scene.addChild( path3 );
    scene.layerAudit();
    
    equal( scene.layers.length, 1, 'canvas only' );
    
    path3.renderer = 'svg';
    scene.layerAudit();
    
    equal( scene.layers.length, 2, 'canvas, svg' );
    
    path1.renderer = 'svg';
    scene.layerAudit();
    
    equal( scene.layers.length, 3, 'svg, canvas, svg' );
    
    path2.renderer = 'svg';
    scene.layerAudit();
    
    equal( scene.layers.length, 1, 'svg' );
    
    path2.renderer = 'canvas';
    scene.layerAudit();
    
    equal( scene.layers.length, 3, 'svg, canvas, svg (again)' );
    
    path1.renderer = 'canvas';
    scene.layerAudit();
    
    equal( scene.layers.length, 2, 'canvas, svg (again)' );
    
    path3.renderer = 'canvas';
    scene.layerAudit();
    
    equal( scene.layers.length, 1, 'canvas (again)' );
  } );
})();
