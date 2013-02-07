
(function(){
  
  var main = $( '#main' );
  
  benchmarkTimer.add( 'Canvas creation', function() {
    document.createElement( 'canvas' );
  } );
  
  benchmarkTimer.add( 'Canvas/context creation', function() {
    var canvas = document.createElement( 'canvas' );
    var context = phet.canvas.initCanvas( canvas );
  } );
  
  benchmarkTimer.add( 'Node creation', function() {
    var node = new phet.scene.Node();
  } );
  
  benchmarkTimer.add( 'Node creation with inline parameters', function() {
    var node = new phet.scene.Node( { x: 5, y: 10 } );
  } );
  
  benchmarkTimer.add( 'Node creation with ES5 setters', function() {
    var node = new phet.scene.Node();
    node.x = 5;
    node.y = 10;
  } );
  
  benchmarkTimer.add( 'Node creation with setters', function() {
    var node = new phet.scene.Node();
    node.setX( 5 );
    node.setY( 10 );
  } );
  
  benchmarkTimer.add( 'Node mutation with ES5 setters', function() {
    node.x = 5;
    node.y = 10;
  }, { setup: function() {
    var node = new phet.scene.Node();
  } } );
  
  benchmarkTimer.add( 'Node mutation with setters', function() {
    node.setX( 5 );
    node.setY( 10 );
  }, { setup: function() {
    var node = new phet.scene.Node();
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
