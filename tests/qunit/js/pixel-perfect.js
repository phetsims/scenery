
(function(){
  'use strict';
  
  module( 'Scenery: Pixel Perfect' );
  
  function pixelTest( name, setup, dataURL, threshold ) {
    asyncTest( name, function() {
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
              snapshotEquals( sceneSnapshot, dataURLSnapshot, threshold, name );
              start();
            }, dataURLSnapshot.width, dataURLSnapshot.height );
          } );
        } else {
          setTimeout( process, 40 );
        }
      }
      
      setTimeout( process, 40 );
    } );
  }
  
  pixelTest( 'Rectangle with stroke',
    function( scene ) {
      scene.addChild( new scenery.Path( {
        shape: kite.Shape.rectangle( 8, 8, 48, 48 ),
        fill: '#000000',
        stroke: '#ff0000',
        lineWidth: 2
      } ) );
    }, "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAQUlEQVRYR+3RwQkAMAgDQN1/6LYTlH5EKRfyN5wZzcnm+2EAgdkCK+K0NleBMQMq//QkYAABAgQIECBAgACBrwU2CDoyBu/dqEkAAAAASUVORK5CYII=",
    0
  );
  
  pixelTest( 'Invisible node with rectangles above and below',
    function( scene ) {
      var shape = kite.Shape.rectangle( 0, 0, 30, 30 );
      scene.addChild( new scenery.Path( {
        shape: shape,
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: -10, y: -10
      } ) );
      scene.addChild( new scenery.Path( {
        shape: shape,
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: 10, y: 10,
        visible: false
      } ) );
      scene.addChild( new scenery.Path( {
        shape: shape,
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: 20, y: 20
      } ) );
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAWklEQVRYR+3WsQoAIAhF0ef/f3TZ0NAQWQgR3ajN5HkmTVLxl3J6I5P8xk4rJAACCCCAAAIIIPC5gI+fJhBbwsYqux7gJPXsTx8mvJF6o53aZVYCIIAAAk8KVE6CQBKyrnp4AAAAAElFTkSuQmCC',
    0
  );
  
  pixelTest( 'Invisible node with rectangles above and below - visible children test + children parameter object',
    function( scene ) {
      var shape = kite.Shape.rectangle( 0, 0, 30, 30 );
      var rect = new scenery.Path( {
        shape: shape,
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2
      } );
      scene.addChild( new scenery.Node( {
        children: [ rect ],
        x: -10, y: -10
      } ) );
      scene.addChild( new scenery.Node( {
        children: [ rect ],
        x: 10, y: 10,
        visible: false
      } ) );
      scene.addChild( new scenery.Node( {
        children: [ rect ],
        x: 20, y: 20
      } ) );
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAWklEQVRYR+3WsQoAIAhF0ef/f3TZ0NAQWQgR3ajN5HkmTVLxl3J6I5P8xk4rJAACCCCAAAIIIPC5gI+fJhBbwsYqux7gJPXsTx8mvJF6o53aZVYCIIAAAk8KVE6CQBKyrnp4AAAAAElFTkSuQmCC',
    0
  );
  
})();
