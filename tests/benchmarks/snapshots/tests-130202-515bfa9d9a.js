
var phet = phet || {};
phet.benchmark = phet.benchmark || {};

(function(){
  
  phet.benchmark.createDetachedScene = function( width, height ) {
    width = width || 640;
    height = height || 480;
    
    var main = $( '#main' );
    main.width( 640 );
    main.height( 480 );
    return new scenery.Scene( main );
  }
  
  // manual testing to see if we can do better than benchmark.js
  var scene = phet.benchmark.createDetachedScene( 256, 256 );
  for( var i = 0; i < 200; i++ ) {
    scene.root.addChild( new scenery.Node( {
      shape: scenery.Shape.rectangle( i, ( 7 * i ) % 200, 20, 20 ),
      fill: 'rgba(255,0,0,1)',
      stroke: '#000000'
    } ) );
  }
  var start = new Date;
  for( var i = 0; i < 100; i++ ) {
    scene.root.rotate( Math.sin( i ) );
    scene.updateScene();
  }
  var end = new Date;
  console.log( benchmarkTimer.currentSnapshot.name + '!!!!!!!!: ' + ( end - start ) );
  
  benchmarkTimer.add( 'Rotating Square 100x', function() {
    for( var i = 0; i < 100; i++ ) {
      scene.root.rotate( Math.sin( i ) );
      scene.updateScene();
    }
  }, {
    setup: function() {
      var scene = phet.benchmark.createDetachedScene( 256, 256 );
      scene.root.addChild( new scenery.Node( {
        shape: scenery.Shape.rectangle( 0, 0, 20, 20 ),
        centerX: 128,
        centerY: 128,
        fill: 'rgba(255,0,0,1)'
      } ) );
    }
  } );
  
  benchmarkTimer.add( 'Rotating Many Squares (with stroke) 100x', function() {
    for( var i = 0; i < 100; i++ ) {
      scene.root.rotate( Math.sin( i ) );
      scene.updateScene();
    }
  }, {
    setup: function() {
      var scene = phet.benchmark.createDetachedScene( 256, 256 );
      for( var i = 0; i < 200; i++ ) {
        scene.root.addChild( new scenery.Node( {
          shape: scenery.Shape.rectangle( i, ( 7 * i ) % 200, 20, 20 ),
          fill: 'rgba(255,0,0,1)',
          stroke: '#000000'
        } ) );
      }
    }
  } );
  
  benchmarkTimer.add( 'Square rotating over background squares 100x', function() {
    for( var i = 0; i < 100; i++ ) {
      node.rotate( Math.sin( i ) );
      scene.updateScene();
    }
  }, {
    setup: function() {
      var scene = phet.benchmark.createDetachedScene( 256, 256 );
      for( var i = 0; i < 200; i++ ) {
        scene.root.addChild( new scenery.Node( {
          shape: scenery.Shape.rectangle( i, ( 7 * i ) % 200, 20, 20 ),
          fill: 'rgba(255,0,0,1)',
          stroke: '#000000'
        } ) );
      }
      var node = new scenery.Node( {
        shape: scenery.Shape.rectangle( 0, 0, 20, 20 ),
        centerX: 128,
        centerY: 128,
        fill: 'rgba(255,0,0,1)',
        stroke: '#000000'
      } );
      scene.root.addChild( node );
    }
  } );
  
  benchmarkTimer.add( 'Static updateScene() over background squares 100x', function() {
    for( var i = 0; i < 100; i++ ) {
      scene.updateScene();
    }
  }, {
    setup: function() {
      var scene = phet.benchmark.createDetachedScene( 256, 256 );
      for( var i = 0; i < 200; i++ ) {
        scene.root.addChild( new scenery.Node( {
          shape: scenery.Shape.rectangle( i, ( 7 * i ) % 200, 20, 20 ),
          fill: 'rgba(255,0,0,1)',
          stroke: '#000000'
        } ) );
      }
      var node = new scenery.Node( {
        shape: scenery.Shape.rectangle( 0, 0, 20, 20 ),
        centerX: 128,
        centerY: 128,
        fill: 'rgba(255,0,0,1)',
        stroke: '#000000'
      } );
      scene.root.addChild( node );
      scene.updateScene();
    }
  } );
  
  benchmarkTimer.add( 'Canvas creation', function() {
    document.createElement( 'canvas' );
  } );
  
  benchmarkTimer.add( 'Canvas/context creation', function() {
    var canvas = document.createElement( 'canvas' );
    var context = phet.canvas.initCanvas( canvas );
  } );
  
  benchmarkTimer.add( 'Node creation', function() {
    var node = new scenery.Node();
  } );
  
  benchmarkTimer.add( 'Node creation with inline parameters', function() {
    var node = new scenery.Node( { x: 5, y: 10 } );
  } );
  
  benchmarkTimer.add( 'Node creation with ES5 setters', function() {
    var node = new scenery.Node();
    node.x = 5;
    node.y = 10;
  } );
  
  benchmarkTimer.add( 'Node creation with setters', function() {
    var node = new scenery.Node();
    node.setX( 5 );
    node.setY( 10 );
  } );
  
  benchmarkTimer.add( 'Node mutation with ES5 setters', function() {
    node.x = 5;
    node.y = 10;
  }, { setup: function() {
    var node = new scenery.Node();
  } } );
  
  benchmarkTimer.add( 'Node mutation with setters', function() {
    node.setX( 5 );
    node.setY( 10 );
  }, { setup: function() {
    var node = new scenery.Node();
  } } );
  
  // benchmarkTimer.add( 'Fast on current version', function() {
  //   var count = 0;
  //   for ( var i = 0; i < 100; i++ ) {
  //     count = count * i + Math.sin( i );
  //   }
  // } );
  
  // benchmarkTimer.add( 'Slow on current version', function() {
    
  // } );
  
  // benchmarkTimer.add( 'Fast deferred on current version', function( deferrer ) {
  //   if ( !deferrer ) {
  //     console.log( 'no deferrer: ' + deferrer );
  //     console.log( 'fast old' );
  //   }
  //   setTimeout( function() {
  //     deferrer.resolve();
  //   }, 1000 );
  // }, { defer: true } );
  
  // benchmarkTimer.add( 'Slow deferred on current version', function( deferrer ) {
  //   if ( !deferrer ) {
  //     console.log( 'no deferrer: ' + deferrer );
  //     console.log( 'slow old' );
  //   }
  //   deferrer.resolve();
  // }, { defer: true } );
  
  // benchmarkTimer.add( 'Control Bench A', function() {
  //   var count = 0;
  //   for ( var i = 0; i < 100; i++ ) {
  //     count = count * i + Math.sin( i );
  //   }
  // } );
  
  // benchmarkTimer.add( 'Control Bench B', function() {
  //   var count = 0;
  //   for ( var i = 0; i < 100; i++ ) {
  //     count = count * i + Math.sin( i );
  //   }
  // }, {
  //   setup: function() {
  //     var count = 0;
  //     for ( var i = 0; i < 10000; i++ ) {
  //       count = count * i + Math.sin( i );
  //     }
  //   },
    
  //   teardown: function() {
  //     var count = 0;
  //     for ( var i = 0; i < 10000; i++ ) {
  //       count = count * i + Math.sin( i );
  //     }
  //   }
  // } );
  
})();
