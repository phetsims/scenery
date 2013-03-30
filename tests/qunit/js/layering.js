
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
})();
