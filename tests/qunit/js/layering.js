
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
  
  test( 'unnamed break #1', function() {
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
  
  test( 'unnamed break #2', function() {
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
    path15.renderer = 'canvas';
    path8.renderer = 'svg';
    path9.insertChild( 0, path14 );
    scene.renderer = 'canvas';
    path16.renderer = null;
    path16.renderer = null;
    path11.renderer = null;
    path16.insertChild( 0, node4 );
    path16.removeChild( node4 );
    path9.insertChild( 0, scene );
    path15.renderer = 'svg';
    node5.renderer = 'svg';
    node4.renderer = null;
    path9.insertChild( 0, path15 );
    path17.renderer = 'svg';
    node7.renderer = 'canvas';
    path9.insertChild( 0, path13 );
    path11.renderer = 'svg';
    path9.renderer = 'svg';
    path8.renderer = null;
    path14.insertChild( 0, path17 );
    path14.renderer = 'svg';
    node4.renderer = null;
    node6.renderer = 'svg';
    path10.renderer = 'canvas';
    path17.insertChild( 0, path13 );
    scene.insertChild( 0, path11 );
    path10.insertChild( 0, path17 );
    scene.removeChild( path11 );
    scene.insertChild( 0, node6 );
    node7.insertChild( 0, path11 );
    path10.removeChild( path17 );
    path14.renderer = null;
    path9.removeChild( scene );
    path11.renderer = 'svg';
    path14.insertChild( 1, node5 );
    path15.insertChild( 0, path17 );
    path10.renderer = null;
    path13.renderer = null;
    path14.insertChild( 2, path11 );
    path12.renderer = null;
    scene.renderer = 'svg';
    node5.renderer = 'svg';
    node3.renderer = null;
    path12.renderer = null;
    node5.renderer = null;
    path17.renderer = null;
    path10.renderer = null;
    path13.insertChild( 0, path12 );
    node5.insertChild( 0, path11 );
    path14.renderer = 'svg';
    path11.renderer = 'canvas';
    path10.insertChild( 0, path9 );
    path11.renderer = 'svg';
    path17.renderer = null;
    path17.insertChild( 1, node6 );
    path15.insertChild( 1, path13 );
    path10.removeChild( path9 );
    path16.insertChild( 0, path10 );
    path14.insertChild( 1, path16 );
    node7.renderer = 'canvas';
    node7.insertChild( 1, path14 );
    path13.removeChild( path12 );
    path15.renderer = 'svg';
    path14.renderer = null;
    path9.insertChild( 2, path8 );
    path16.renderer = null;
    node3.insertChild( 0, path14 );
    path13.renderer = 'svg';
    node5.renderer = null;
    node3.removeChild( path14 );
    path15.insertChild( 0, node4 );
    path17.insertChild( 1, node5 );
    path14.renderer = 'canvas';
    path9.renderer = 'svg';
    path14.insertChild( 2, path13 );
    node7.removeChild( path11 );
    scene.insertChild( 1, node5 );
    path8.renderer = null;
    node7.removeChild( path14 );
    node4.insertChild( 0, path16 );
    path10.renderer = 'svg';
    path13.renderer = 'canvas';
    scene.renderer = 'canvas';
    path13.renderer = 'svg';
    path15.renderer = 'svg';
    path10.renderer = null;
    path16.renderer = 'svg';
    node5.insertChild( 1, path8 );
    path12.renderer = null;
    path12.renderer = 'canvas';
    node6.insertChild( 0, node4 );
    path14.insertChild( 4, node6 );
    scene.renderer = null;
    path16.renderer = 'canvas';
    path15.insertChild( 0, scene );
    path12.insertChild( 0, path11 );
    path13.renderer = 'svg';
    path14.insertChild( 1, scene );
    path9.insertChild( 4, node4 );
    node6.insertChild( 0, path13 );
    node4.renderer = null;
    node4.removeChild( path16 );
    path16.insertChild( 1, node7 );
    node7.renderer = 'canvas';
    scene.insertChild( 2, node3 );
    scene.insertChild( 1, path16 );
    
    expect( 0 );
  } );
  
  test( 'unnamed break #3', function() {
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
    path12.renderer = null;
    path16.insertChild( 0, node5 );
    scene.renderer = 'canvas';
    scene.renderer = 'svg';
    path13.insertChild( 0, node6 );
    path13.insertChild( 1, path17 );
    node5.renderer = null;
    scene.insertChild( 0, path12 );
    node7.insertChild( 0, path15 );
    path13.renderer = null;
    path12.renderer = null;
    path12.insertChild( 0, node4 );
    path13.removeChild( node6 );
    path9.renderer = 'svg';
    node7.insertChild( 0, node5 );
    path13.removeChild( path17 );
    path10.renderer = null;
    path12.renderer = 'canvas';
    node3.insertChild( 0, path11 );
    path15.insertChild( 0, node5 );
    node6.insertChild( 0, path16 );
    path16.renderer = null;
    path9.renderer = 'canvas';
    node7.renderer = 'canvas';
    node7.removeChild( path15 );
    node6.insertChild( 0, path8 );
    path15.renderer = null;
    path11.renderer = null;
    path16.renderer = null;
    node7.removeChild( node5 );
    path12.insertChild( 0, path13 );
    node4.renderer = null;
    path9.renderer = null;
    node6.removeChild( path8 );
    path11.insertChild( 0, path9 );
    path12.renderer = null;
    path8.insertChild( 0, path16 );
    path10.insertChild( 0, node4 );
    path10.renderer = 'svg';
    path15.removeChild( node5 );
    path11.removeChild( path9 );
    path14.renderer = null;
    path8.renderer = null;
    node4.renderer = null;
    path16.renderer = null;
    path10.insertChild( 0, node7 );
    path8.insertChild( 0, node3 );
    scene.removeChild( path12 );
    path13.renderer = 'canvas';
    path11.insertChild( 0, path10 );
    path11.insertChild( 1, path15 );
    node5.renderer = null;
    path15.renderer = 'svg';
    path15.renderer = 'canvas';
    path12.insertChild( 1, path10 );
    path15.renderer = null;
    path10.insertChild( 1, path13 );
    node3.insertChild( 0, scene );
    path14.insertChild( 0, node3 );
    node6.insertChild( 1, path11 );
    path9.renderer = null;
    path16.renderer = 'canvas';
    scene.insertChild( 0, path10 );
    node5.renderer = null;
    node3.renderer = null;
    path12.removeChild( path13 );
    path12.insertChild( 2, node3 );
    node3.renderer = null;
    node4.insertChild( 0, node5 );
    path13.insertChild( 0, node5 );
    path13.insertChild( 0, path15 );
    path15.renderer = 'svg';
    path11.renderer = null;
    path15.renderer = 'canvas';
    path11.removeChild( path10 );
    path13.renderer = null;
    path10.renderer = null;
    scene.renderer = null;
    path13.renderer = null;
    node7.renderer = null;
    path13.renderer = null;
    node5.renderer = null;
    path12.renderer = null;
    node4.removeChild( node5 );
    path16.renderer = 'svg';
    path8.insertChild( 0, path17 );
    path9.insertChild( 0, path15 );
    node3.renderer = null;
    node6.insertChild( 0, path14 );
    path16.insertChild( 0, path12 );
    path12.insertChild( 3, path14 );
    node5.renderer = 'canvas';
    path10.renderer = 'canvas';
    path11.removeChild( path15 );
    node4.renderer = 'svg';
    path11.renderer = null;
    path8.renderer = 'canvas';
    node7.insertChild( 0, node5 );
    path14.renderer = null;
    path8.removeChild( node3 );
    path16.removeChild( path12 );
    node3.removeChild( scene );
    node4.renderer = null;
    node6.renderer = null;
    path12.renderer = 'svg';
    path16.renderer = 'svg';
    path14.removeChild( node3 );
    path16.renderer = 'canvas';
    node6.removeChild( path11 );
    path16.removeChild( node5 );
    path8.renderer = 'canvas';
    scene.renderer = null;
    path9.removeChild( path15 );
    node4.renderer = null;
    path17.insertChild( 0, scene );
    node6.removeChild( path16 );
    path14.renderer = null;
    path8.removeChild( path17 );
    path12.removeChild( node4 );
    path14.renderer = 'svg';
    scene.removeChild( path10 );
    node3.insertChild( 1, node5 );
    path12.renderer = 'canvas';
    node6.removeChild( path14 );
    node3.renderer = null;
    path14.renderer = null;
    scene.insertChild( 0, node5 );
    path10.insertChild( 2, scene );
    path10.insertChild( 2, path14 );
    path10.removeChild( path13 );
    path10.removeChild( scene );
    node6.insertChild( 0, path15 );
    path12.insertChild( 2, path11 );
    path15.renderer = 'canvas';
    path11.insertChild( 0, node6 );
    node5.renderer = null;
    path16.insertChild( 0, path17 );
    scene.removeChild( node5 );
    node6.removeChild( path15 );
    path10.renderer = 'canvas';
    path12.insertChild( 3, path16 );
    node7.renderer = null;
    path15.insertChild( 0, path17 );
    path14.renderer = 'canvas';
    path14.insertChild( 0, node6 );
    path10.removeChild( node7 );
    path12.insertChild( 3, node5 );
    node7.insertChild( 0, path13 );
    node7.insertChild( 0, node3 );
    path10.removeChild( node4 );
    path15.removeChild( path17 );
    node5.insertChild( 0, path17 );
    path11.insertChild( 0, scene );
    node7.insertChild( 2, path17 );
    path8.renderer = null;
    path14.removeChild( node6 );
    path13.renderer = 'canvas';
    node7.insertChild( 1, node6 );
    path16.renderer = null;
    path9.renderer = 'svg';
    path13.insertChild( 0, path16 );
    node5.removeChild( path17 );
    node7.insertChild( 0, path14 );
    node5.insertChild( 0, scene );
    node5.renderer = 'svg';
    path10.insertChild( 0, node5 );
    path17.renderer = 'svg';
    path12.removeChild( path10 );
    node3.removeChild( node5 );
    path16.insertChild( 0, node4 );
    path15.insertChild( 0, path17 );
    path8.removeChild( path16 );
    path17.renderer = 'canvas';
    node7.insertChild( 3, path8 );
    path15.removeChild( path17 );
    path9.renderer = 'svg';
    node3.renderer = null;
    node5.insertChild( 0, path9 );
    path8.insertChild( 0, path10 );
    node3.insertChild( 0, node4 );
    path17.removeChild( scene );
    path10.renderer = 'svg';
    path11.renderer = null;
    scene.insertChild( 0, path9 );
    path16.removeChild( path17 );
    node7.renderer = null;
    path12.removeChild( node3 );
    path8.renderer = null;
    path13.insertChild( 3, node4 );
    path12.removeChild( node5 );
    node7.removeChild( node3 );
    node3.removeChild( node4 );
    scene.removeChild( path9 );
    path12.renderer = 'canvas';
    node7.removeChild( node5 );
    path8.removeChild( path10 );
    node5.insertChild( 1, node4 );
    path9.renderer = 'canvas';
    path9.insertChild( 0, path14 );
    node5.insertChild( 3, node3 );
    path11.insertChild( 1, node4 );
    path8.insertChild( 0, path17 );
    path12.removeChild( path11 );
    path14.renderer = null;
    path16.removeChild( node4 );
    node4.renderer = null;
    node3.renderer = 'svg';
    path8.insertChild( 0, path13 );
    path17.insertChild( 0, node4 );
    scene.insertChild( 0, path15 );
    path15.insertChild( 0, node6 );
    node6.renderer = 'svg';
    node3.renderer = 'svg';
    node5.insertChild( 1, path17 );
    path14.renderer = null;
    node3.renderer = null;
    path17.removeChild( node4 );
    path17.insertChild( 0, path11 );
    scene.insertChild( 0, path14 );
    
    expect( 0 );
  } );
  
  test( 'unnamed break #4', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    var path2 = new scenery.Path( {
      renderer: 'svg'
    } )
    var node3 = new scenery.Node( {} )
    node3.addChild( path2 );
    var path4 = new scenery.Path( {} )
    path4.addChild( node3 );
    var node5 = new scenery.Node( {} )
    node5.addChild( path4 );
    var path6 = new scenery.Path( {} )
    path6.addChild( node5 );
    var node7 = new scenery.Node( {
      renderer: 'svg'
    } )
    node7.addChild( node3 );
    node7.addChild( path6 );
    scene.mutate( {
      renderer: 'svg'
    } )
    scene.addChild( node7 );
    scene.addChild( path6 );
    var path8 = new scenery.Path( {} )
    path8.addChild( path6 );
    var path9 = new scenery.Path( {
      renderer: 'svg'
    } )
    path9.addChild( path2 );
    path9.addChild( path8 );
    var node10 = new scenery.Node( {} )
    node10.addChild( path9 );
    var node11 = new scenery.Node( {
      renderer: 'canvas'
    } )
    node11.addChild( path2 );
    var path12 = new scenery.Path( {} )
    path12.addChild( node11 );
    path12.addChild( node3 );
    path12.addChild( path8 );
    var path13 = new scenery.Path( {
      renderer: 'svg'
    } )
    path13.addChild( path12 );
    path13.addChild( node11 );
    var path14 = new scenery.Path( {
      renderer: 'canvas'
    } )
    path14.addChild( node10 );
    path14.addChild( path13 );
    var path15 = new scenery.Path( {} )
    path15.addChild( node3 );
    
    // causes the break
    path4.renderer = 'canvas';
    
    expect( 0 );
  } );
  
  test( 'unnamed break #5 (layer ordering)', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    var path10 = new scenery.Path( {
      renderer: 'svg'
    } )
    var path14 = new scenery.Path( {} )
    var path12 = new scenery.Path( {} )
    path12.addChild( path10 );
    path12.addChild( path14 );
    var node4 = new scenery.Node( {
      renderer: 'canvas'
    } )
    node4.addChild( path12 );
    var path11 = new scenery.Path( {
      renderer: 'svg'
    } )
    path11.addChild( path10 );
    var path16 = new scenery.Path( {} )
    path16.addChild( path11 );
    var node5 = new scenery.Node( {} )
    var node7 = new scenery.Node( {
      renderer: 'svg'
    } )
    node7.addChild( node4 );
    node7.addChild( path16 );
    node7.addChild( path10 );
    node7.addChild( node5 );
    var path15 = new scenery.Path( {} )
    scene.mutate( {} )
    scene.addChild( node7 );
    scene.addChild( path15 );
    scene.addChild( node5 );
    var node6 = new scenery.Node( {} )
    node6.addChild( path15 );
    node6.addChild( node7 );
    var path17 = new scenery.Path( {} )
    path17.addChild( node6 );
    var path13 = new scenery.Path( {} )
    path13.addChild( path10 );
    path13.addChild( path11 );
    path13.addChild( path17 );
    var path8 = new scenery.Path( {} )
    path8.addChild( path12 );
    path8.addChild( path11 );
    var node3 = new scenery.Node( {
      renderer: 'canvas'
    } )
    node3.addChild( node4 );
    var path9 = new scenery.Path( {} )
    path9.addChild( path11 );
    path9.addChild( path10 );
    
    // breaks
    node7.renderer = null;
    
    expect( 0 );
  } );
  
  test( 'unnamed break #6 (layer split)', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    var node6 = new scenery.Node( {} )
    var path16 = new scenery.Path( {
      renderer: 'canvas'
    } )
    var node7 = new scenery.Node( {
      layerSplitBefore: true,
      layerSplitAfter: true
    } )
    var node5 = new scenery.Node( {
      renderer: 'svg'
    } )
    node5.addChild( node6 );
    node5.addChild( path16 );
    node5.addChild( node7 );
    var node4 = new scenery.Node( {
      renderer: 'svg'
    } )
    node4.addChild( node5 );
    var path12 = new scenery.Path( {} )
    var path14 = new scenery.Path( {
      layerSplitBefore: true
    } )
    var path9 = new scenery.Path( {
      renderer: 'svg',
      layerSplitAfter: true
    } )
    path9.addChild( path12 );
    path9.addChild( path14 );
    scene.mutate( {} )
    var path13 = new scenery.Path( {
      renderer: 'canvas'
    } )
    path13.addChild( node7 );
    var path10 = new scenery.Path( {
      renderer: 'canvas',
      layerSplitBefore: true
    } )
    path10.addChild( scene );
    path10.addChild( path12 );
    path10.addChild( path13 );
    var path8 = new scenery.Path( {
      renderer: 'canvas'
    } )
    path8.addChild( node4 );
    path8.addChild( path9 );
    path8.addChild( node6 );
    path8.addChild( path10 );
    var node3 = new scenery.Node( {
      renderer: 'canvas'
    } )
    node3.addChild( path8 );
    node3.addChild( path13 );
    node3.addChild( path9 );
    node3.addChild( node4 );
    node3.addChild( path10 );
    node3.addChild( path12 );
    var path17 = new scenery.Path( {
      renderer: 'svg'
    } )
    path17.addChild( path16 );
    path17.addChild( node6 );
    path17.addChild( node3 );
    path17.addChild( path13 );
    path17.addChild( scene );
    var path15 = new scenery.Path( {} )
    path15.addChild( path14 );
    var path11 = new scenery.Path( {
      renderer: 'canvas',
      layerSplitAfter: true
    } )
    path11.addChild( path13 );
    path11.addChild( path9 );
    path11.addChild( node6 );
    path11.addChild( scene );
    path11.addChild( path15 );
    path11.addChild( path12 );
    path11.addChild( node5 );
    
    // breaks
    scene.insertChild( 0, path9 );
    
    expect( 0 );
  } );
})();
