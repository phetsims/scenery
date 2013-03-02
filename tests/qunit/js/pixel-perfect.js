
(function(){
  'use strict';
  
  module( 'Pixel Perfect' );
  
  asyncTest( 'Rectangle with stroke', function() {
    function setup( scene ) {
      scene.addChild( new scenery.Path( {
        shape: scenery.Shape.rectangle( 8, 8, 48, 48 ),
        fill: '#000000',
        stroke: '#ff0000',
        lineWidth: 2
      } ) );
    }
    
    var dataURL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAQUlEQVRYR+3RwQkAMAgDQN1/6LYTlH5EKRfyN5wZzcnm+2EAgdkCK+K0NleBMQMq//QkYAABAgQIECBAgACBrwU2CDoyBu/dqEkAAAAASUVORK5CYII=";
    
    var name = QUnit.config.current.testName;
    
    function process() {
      if ( window.snapshotFromDataURL ) {
        snapshotFromDataURL( dataURL, function( dataURLSnapshot ) {
          var $div = $( '<div>' );
          $div.width( dataURLSnapshot.width );
          $div.height( dataURLSnapshot.height );
          
          var scene = new scenery.Scene( $div );
          setup( scene );
          scene.updateScene();
          
          asyncSnapshot( scene, function( sceneSnapshot ) {
            snapshotEquals( sceneSnapshot, dataURLSnapshot, 0, name );
            start();
          }, dataURLSnapshot.width, dataURLSnapshot.height );
        } );
      } else {
        setTimeout( process, 40 );
      }
    }
    
    setTimeout( process, 40 );
  } );
})();
