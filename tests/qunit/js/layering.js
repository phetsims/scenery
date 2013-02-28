
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
    
    var d = new scenery.Path();
    b.addChild( d );
    
    equal( scene.layers.length, 1, 'still just a single layer' );
    
    b.renderer = 'canvas';
    
    equal( scene.layers.length, 1, 'scene is canvas, so b should not trigger any more layers' );
  } );
})();
