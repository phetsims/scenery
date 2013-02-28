
(function(){
  'use strict';
  
  module( 'Layering' );
  
  test( 'Layer quantity check', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    equal( scene.layers.length, 0, 'no layers at the start' );
    
    var a = new scenery.Path();
    var b = new scenery.Path();
    var c = new scenery.Path();
    scene.addChild( a );
    scene.addChild( b );
    scene.addChild( c );
    
    equal( scene.layers.length, 1, 'just a single layer for three paths' );
    equal( scene.layerLookup( a.getUniqueTrail() ), scene.layers[0] );
    equal( scene.layerLookup( b.getUniqueTrail() ), scene.layers[0] );
    equal( scene.layerLookup( c.getUniqueTrail() ), scene.layers[0] );
    
    var d = new scenery.Path();
    b.addChild( d );
    
    equal( scene.layers.length, 1, 'still just a single layer' );
    equal( scene.layerLookup( a.getUniqueTrail() ), scene.layers[0] );
    equal( scene.layerLookup( b.getUniqueTrail() ), scene.layers[0] );
    equal( scene.layerLookup( c.getUniqueTrail() ), scene.layers[0] );
    
    b.renderer = 'canvas';
    
    equal( scene.layers.length, 1, 'scene is canvas, so b should not trigger any more layers' );
    equal( scene.layerLookup( a.getUniqueTrail() ), scene.layers[0] );
    equal( scene.layerLookup( b.getUniqueTrail() ), scene.layers[0] );
    equal( scene.layerLookup( c.getUniqueTrail() ), scene.layers[0] );
    
    b.renderer = 'svg';
    
    equal( scene.layers.length, 3, 'should be canvas, svg, canvas' );
    equal( scene.layerLookup( a.getUniqueTrail() ), scene.layers[0] );
    equal( scene.layerLookup( b.getUniqueTrail() ), scene.layers[1] );
    equal( scene.layerLookup( c.getUniqueTrail() ), scene.layers[2] );
    
    c.renderer = 'svg';
    
    equal( scene.layers.length, 2, 'should be canvas, svg (combined)' );
    equal( scene.layerLookup( a.getUniqueTrail() ), scene.layers[0] );
    equal( scene.layerLookup( b.getUniqueTrail() ), scene.layers[1] );
    equal( scene.layerLookup( c.getUniqueTrail() ), scene.layers[1] );
    
    // b.rendererOptions = {
    //   someUniqueThingToThisLayer: 5
    // };
    
    // equal( scene.layers.length, 3, 'should be canvas, svg (with options), svg' );
    // equal( scene.layerLookup( a.getUniqueTrail() ), scene.layers[0] );
    // equal( scene.layerLookup( b.getUniqueTrail() ), scene.layers[1] );
    // equal( scene.layerLookup( c.getUniqueTrail() ), scene.layers[2] );
    
    // window.debugScene = scene;
  } );
})();
