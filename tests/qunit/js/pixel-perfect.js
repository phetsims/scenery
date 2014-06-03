
(function(){
  'use strict';
  
  module( 'Scenery: Pixel Perfect' );
  
  function pixelTest( name, setup, dataURL, threshold ) {
    asyncTest( name, function() {
      function process() {
        if ( window.snapshotFromDataURL ) {
          snapshotFromDataURL( dataURL, function( dataURLSnapshot ) {
            var scene = new scenery.Node();
            setup( scene );
            
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
      scene.addChild( new scenery.Path( kite.Shape.rectangle( 8, 8, 48, 48 ), {
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
      scene.addChild( new scenery.Path( shape, {
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: -10, y: -10
      } ) );
      scene.addChild( new scenery.Path( shape, {
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: 10, y: 10,
        visible: false
      } ) );
      scene.addChild( new scenery.Path( shape, {
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
      var rect = new scenery.Path( shape, {
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
  
  pixelTest( 'Invisible => Visible',
    function( scene ) {
      var rect = new scenery.Rectangle( 0, 0, 16, 16, { fill: '#f00', visible: false } );
      scene.addChild( rect );
      rect.visible = true;
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAQUlEQVRYR+3WMQoAMAgDwPj/R7e+wQpdTrIK4Ratk3TmU0lnPqUAAQIECBAg8F1gfsl3Np+eiY0KChAgQIAAAQIX6VUgIfXDabwAAAAASUVORK5CYII=',
    0
  );
  
  pixelTest( 'Color changes',
    function( scene ) {
      var color = new scenery.Color( '#f00' );
      var rect = new scenery.Rectangle( 0, 0, 16, 16, { fill: color } );
      scene.addChild( rect );
      color.blue = 255;
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAP0lEQVRYR+3WsQ0AQAgCQNl/aH8ITL45exJyDWZnd4rLJEV8ogABAgQIECDwXaDZ8ots9UwoQIAAAQIECFwIPEjvMCG7TeaeAAAAAElFTkSuQmCC',
    0
  );
  
  pixelTest( 'Invisible repaints',
    function( scene ) {
      var rect = new scenery.Rectangle( 0, 0, 16, 16, { fill: '#f00' } );
      scene.addChild( rect );
      rect.visible = false;
      rect.x = 16;
      rect.visible = true;
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAQUlEQVRYR+3WMQoAMAgDQP3/o9tCf2AGlxPXQLhFu8I5VW/n0/PoTypAgAABAgQIrAuk5zz+BxQgQIAAAQIEUoELbg4gAWKut4YAAAAASUVORK5CYII=',
    0
  );
  
  pixelTest( 'Children under invisible made visible',
    function( scene ) {
      var rect = new scenery.Rectangle( 0, 0, 16, 16, { fill: '#f00' } );
      scene.addChild( rect );
      rect.visible = false;
      rect.x = 16;
      rect.addChild( new scenery.Rectangle( -16, 0, 16, 16, { fill: '#00f' } ) );
      rect.visible = true;
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAARUlEQVRYR+3WsQ0AMAgDsPT/o6naFzKwmD0i8gInmUkxk1Ok89IKECBAgAABAusC1Tmvw903Ua//D8nuKECAAAECBNYFLsnVMAEcXLUQAAAAAElFTkSuQmCC',
    0
  );
  
  pixelTest( 'Children under invisible made invisible',
    function( scene ) {
      var rect = new scenery.Rectangle( 0, 0, 16, 16, { fill: '#f00' } );
      scene.addChild( rect );
      rect.visible = false;
      rect.x = 16;
      rect.addChild( new scenery.Rectangle( -16, 0, 16, 16, { fill: '#00f', visible: false } ) );
      rect.visible = true;
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAQUlEQVRYR+3WMQoAMAgDQP3/o9tCf2AGlxPXQLhFu8I5VW/n0/PoTypAgAABAgQIrAuk5zz+BxQgQIAAAQIEUoELbg4gAWKut4YAAAAASUVORK5CYII=',
    0
  );
  
  pixelTest( 'SVG: Rectangle',
    function( scene ) {
      var rect = new scenery.Rectangle( 0, 0, 16, 16, { fill: '#f00', renderer: 'svg' } );
      scene.addChild( rect );
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAQUlEQVRYR+3WMQoAMAgDwPj/R7e+wQpdTrIK4Ratk3TmU0lnPqUAAQIECBAg8F1gfsl3Np+eiY0KChAgQIAAAQIX6VUgIfXDabwAAAAASUVORK5CYII=',
    0
  );
  
  pixelTest( 'SVG: Invisible Rectangle',
    function( scene ) {
      var rect = new scenery.Rectangle( 0, 0, 16, 16, { fill: '#f00', renderer: 'svg', visible: false } );
      scene.addChild( rect );
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAALUlEQVRYR+3QQREAAAABQfqXFsNnFTizzXk99+MAAQIECBAgQIAAAQIECBAgMBo/ACHo7lH9AAAAAElFTkSuQmCC',
    0
  );
})();
