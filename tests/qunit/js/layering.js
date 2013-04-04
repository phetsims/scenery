
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
