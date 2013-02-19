

(function(){
  "use strict";
  
  // $( '#display' ).hide();
  
  var includeBleedingEdgeCanvasTests = false;
  
  var canvasWidth = 320;
  var canvasHeight = 240;
  
  var unicodeTestStrings = [
    "This is a test",
    "Newline\nJaggies?",
    "\u222b",
    "\ufdfa",
    "\u00a7",
    "\u00C1",
    "\u00FF",
    "\u03A9",
    "\u0906",
    "\u79C1",
    "\u9054",
    "A\u030a\u0352\u0333\u0325\u0353\u035a\u035e\u035e",
    "0\u0489",
    "\u2588"
  ];
  
  // takes a snapshot of a scene and stores the pixel data, so that we can compare them
  function snapshot( scene ) {
    var canvas = document.createElement( 'canvas' );
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    var context = phet.canvas.initCanvas( canvas );
    scene.renderToCanvas( canvas, context );
    var data = context.getImageData( 0, 0, canvasWidth, canvasHeight );
    return data;
  }
  
  function snapshotToCanvas( snapshot ) {
    var canvas = document.createElement( 'canvas' );
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    var context = phet.canvas.initCanvas( canvas );
    context.putImageData( snapshot, 0, 0 );
    $( canvas ).css( 'border', '1px solid black' );
    return canvas;
  }
  
  // compares two pixel snapshots and uses the qunit's assert to verify they are the same
  function snapshotEquals( a, b, threshold, message ) {
    var isEqual = a.width == b.width && a.height == b.height;
    var largestDifference = 0;
    if ( isEqual ) {
      for ( var i = 0; i < a.data.length; i++ ) {
        if ( a.data[i] != b.data[i] && Math.abs( a.data[i] - b.data[i] ) > threshold ) {
          // console.log( message + ": " + Math.abs( a.data[i] - b.data[i] ) );
          largestDifference = Math.max( largestDifference, Math.abs( a.data[i] - b.data[i] ) );
          isEqual = false;
          // break;
        }
      }
    }
    if ( largestDifference > 0 ) {
      var display = $( '#display' );
      // header
      var note = document.createElement( 'h2' );
      $( note ).text( message );
      display.append( note );
      var differenceDiv = document.createElement( 'div' );
      $( differenceDiv ).text( 'Largest pixel color-channel difference: ' + largestDifference );
      display.append( differenceDiv );
      
      display.append( snapshotToCanvas( a ) );
      display.append( snapshotToCanvas( b ) );
      
      // for a line-break
      display.append( document.createElement( 'div' ) );
      
    }
    ok( isEqual, message );
    return isEqual;
  }
  
  // compares the "update" render against a full render in-between a series of steps
  function updateVsFullRender( actions ) {
    var mainScene = new scenery.Scene( $( '#main' ) );
    var secondaryScene = new scenery.Scene( $( '#secondary' ) );
    
    for ( var i = 0; i < actions.length; i++ ) {
      var action = actions[i];
      action( mainScene );
      mainScene.updateScene();
      
      secondaryScene.dispose();
      secondaryScene = new scenery.Scene( $( '#secondary' ) );
      for ( var j = 0; j <= i; j++ ) {
        actions[j]( secondaryScene );
      }
      secondaryScene.updateScene();
      
      var isEqual = snapshotEquals( snapshot( mainScene ), snapshot( secondaryScene ), 0, 'action #' + i );
      if ( !isEqual ) {
        break;
      }
    }
  }
  
  function sceneEquals( constructionA, constructionB, message, threshold ) {
    if ( threshold === undefined ) {
      threshold = 0;
    }
    
    var sceneA = new scenery.Scene( $( '#main' ) );
    var sceneB = new scenery.Scene( $( '#secondary' ) );
    
    constructionA( sceneA );
    constructionB( sceneB );
    
    sceneA.renderScene();
    sceneB.renderScene();
    
    var isEqual = snapshotEquals( snapshot( sceneA ), snapshot( sceneB ), threshold, message );
    
    // TODO: consider showing if tests fail
    return isEqual;
  }
  
  function strokeEqualsFill( shapeToStroke, shapeToFill, strokeNodeSetup, message ) {
    sceneEquals( function( scene ) {
      var node = new scenery.Path();
      node.setShape( shapeToStroke );
      node.setStroke( '#000000' );
      if ( strokeNodeSetup ) { strokeNodeSetup( node ); }
      scene.addChild( node );
    }, function( scene ) {
      var node = new scenery.Path();
      node.setShape( shapeToFill );
      node.setFill( '#000000' );
      // node.setStroke( '#ff0000' ); // for debugging strokes
      scene.addChild( node );
      // node.validateBounds();
      // scene.addChild( new scenery.Path( {
      //   shape: scenery.Shape.bounds( node.getSelfBounds() ),
      //   fill: 'rgba(0,0,255,0.5)'
      // } ) );
    }, message, 128 ); // threshold of 128 due to antialiasing differences between fill and stroke... :(
  }
  
  function compareShapeBackends( shape, message ) {
    
  }
  
  function testTextBounds( getBoundsOfText, fontDrawingStyles, message ) {
    var precision = 1;
    var title = document.createElement( 'h2' );
    $( title ).text( message );
    $( '#display' ).append( title );
    _.each( unicodeTestStrings, function( testString ) {
      var testBounds = getBoundsOfText( testString, fontDrawingStyles );
      var bestBounds = scenery.canvasTextBoundsAccurate( testString, fontDrawingStyles );
      
      var widthOk = Math.abs( testBounds.width() - bestBounds.width() ) < precision;
      var heightOk = Math.abs( testBounds.height() - bestBounds.height() ) < precision;
      var xOk = Math.abs( testBounds.x() - bestBounds.x() ) < precision;
      var yOk = Math.abs( testBounds.y() - bestBounds.y() ) < precision;
      
      var allOk = widthOk && heightOk && xOk && yOk;
      
      ok( widthOk, testString + ' width error: ' + Math.abs( testBounds.width() - bestBounds.width() ) );
      ok( heightOk, testString + ' height error: ' + Math.abs( testBounds.height() - bestBounds.height() ) );
      ok( xOk, testString + ' x error: ' + Math.abs( testBounds.x() - bestBounds.x() ) );
      ok( yOk, testString + ' y error: ' + Math.abs( testBounds.y() - bestBounds.y() ) );
      
      // show any failures
      var pad = 5;
      var scaling = 4; // scale it for display accuracy
      var canvas = document.createElement( 'canvas' );
      canvas.width = Math.ceil( bestBounds.width() + pad * 2 ) * scaling;
      canvas.height = Math.ceil( bestBounds.height() + pad * 2 ) * scaling;
      var context = phet.canvas.initCanvas( canvas );
      context.scale( scaling, scaling );
      context.translate( pad - bestBounds.x(), pad - bestBounds.y() ); // center the text in our bounds
      
      // background bounds
      context.fillStyle = allOk ? '#ccffcc' : '#ffcccc'; // red/green depending on whether it passed
      context.fillRect( testBounds.x(), testBounds.y(), testBounds.width(), testBounds.height() );
      
      // text on top
      context.fillStyle = 'rgba(0,0,0,0.7)';
      context.font = fontDrawingStyles.font;
      context.textAlign = fontDrawingStyles.textAlign;
      context.textBaseline = fontDrawingStyles.textBaseline;
      context.direction = fontDrawingStyles.direction;
      context.fillText( testString, 0, 0 );
      
      $( canvas ).css( 'border', '1px solid black' );
      $( '#display' ).append( canvas );
    } );
  }
  
  function equalsApprox( a, b, message ) {
    ok( Math.abs( a - b ) < 0.0000001, ( message ? message + ': ' : '' ) + a + ' =? ' + b );
  }
  
  function createTestNodeTree() {
    var node = new scenery.Node();
    node.addChild( new scenery.Node() );
    node.addChild( new scenery.Node() );
    node.addChild( new scenery.Node() );
    
    node.children[0].addChild( new scenery.Node() );
    node.children[0].addChild( new scenery.Node() );
    node.children[0].addChild( new scenery.Node() );
    node.children[0].addChild( new scenery.Node() );
    node.children[0].addChild( new scenery.Node() );
    
    node.children[0].children[1].addChild( new scenery.Node() );
    node.children[0].children[3].addChild( new scenery.Node() );
    node.children[0].children[3].addChild( new scenery.Node() );
    
    node.children[0].children[3].children[0].addChild( new scenery.Node() );
    
    return node;
  }
  
  /*---------------------------------------------------------------------------*
  * TESTS BELOW
  *----------------------------------------------------------------------------*/   
  
  module( 'Scene Regression' );
  
  test( 'Dirty bounds propagation test', function() {
    var node = createTestNodeTree();
    
    node.validateBounds();
    
    ok( !node._childBoundsDirty );
    
    node.children[0].children[3].children[0].invalidateBounds();
    
    ok( node._childBoundsDirty );
  } );
  
  test( 'Scene Layer test 1', function() {
    var sceneA = new scenery.Scene( $( '#main' ) );
    var sceneB = new scenery.Scene( $( '#secondary' ) );
    
    var node = new scenery.Path();
    var child = new scenery.Path();
    node.addChild( child );
    
    sceneA.addChild( node );
    sceneB.addChild( child );
    
    sceneA.rebuildLayers();
    
    // console.log( sceneA.layers );
    
    
    var a = new scenery.Scene( $( '#main' ) );
    var b = new scenery.Scene( $( '#secondary' ) );
    var c = new scenery.Node();
    
    b.addChild( c );
    a.addChild( b );
    a.addChild( c );
    
    expect( 0 );
  } );
  
  test( 'Canvas 2D Context and Features', function() {
    var canvas = document.createElement( 'canvas' );
    var context = phet.canvas.initCanvas( canvas );
    
    ok( context, 'context' );
    
    var neededMethods = [
      'arc',
      'arcTo',
      'beginPath',
      'bezierCurveTo',
      'clearRect',
      'clip',
      'closePath',
      'fill',
      'fillRect',
      'fillStyle',
      'isPointInPath',
      'lineTo',
      'moveTo',
      'rect',
      'restore',
      'quadraticCurveTo',
      'save',
      'setTransform',
      'stroke',
      'strokeRect',
      'strokeStyle'
    ];
    _.each( neededMethods, function( method ) {
      ok( context[method] !== undefined, 'context.' + method );
    } );
  } );
  
  test( 'Trail next/previous', function() {
    var node = createTestNodeTree();
    
    // walk it forward
    var trail = new scenery.Trail( [ node ] );
    equal( 1, trail.length );
    trail = trail.next();
    equal( 2, trail.length );
    trail = trail.next();
    equal( 3, trail.length );
    trail = trail.next();
    equal( 3, trail.length );
    trail = trail.next();
    equal( 4, trail.length );
    trail = trail.next();
    equal( 3, trail.length );
    trail = trail.next();
    equal( 3, trail.length );
    trail = trail.next();
    equal( 4, trail.length );
    trail = trail.next();
    equal( 5, trail.length );
    trail = trail.next();
    equal( 4, trail.length );
    trail = trail.next();
    equal( 3, trail.length );
    trail = trail.next();
    equal( 2, trail.length );
    trail = trail.next();
    equal( 2, trail.length );
    
    // make sure walking off the end gives us null
    equal( null, trail.next() );
    
    trail = trail.previous();
    equal( 2, trail.length );
    trail = trail.previous();
    equal( 3, trail.length );
    trail = trail.previous();
    equal( 4, trail.length );
    trail = trail.previous();
    equal( 5, trail.length );
    trail = trail.previous();
    equal( 4, trail.length );
    trail = trail.previous();
    equal( 3, trail.length );
    trail = trail.previous();
    equal( 3, trail.length );
    trail = trail.previous();
    equal( 4, trail.length );
    trail = trail.previous();
    equal( 3, trail.length );
    trail = trail.previous();
    equal( 3, trail.length );
    trail = trail.previous();
    equal( 2, trail.length );
    trail = trail.previous();
    equal( 1, trail.length );
    
    // make sure walking off the start gives us null
    equal( null, trail.previous() );
  } );
  
  test( 'Trail comparison', function() {
    var node = createTestNodeTree();
    
    // get a list of all trails in render order
    var trails = [];
    var currentTrail = new scenery.Trail( node ); // start at the first node
    
    while ( currentTrail ) {
      trails.push( currentTrail );
      currentTrail = currentTrail.next();
    }
    
    equal( 13, trails.length, 'Trail for each node' );
    
    for ( var i = 0; i < trails.length; i++ ) {
      for ( var j = i; j < trails.length; j++ ) {
        var comparison = trails[i].compare( trails[j] );
        
        // make sure that every trail compares as expected (0 and they are equal, -1 and i < j)
        equal( i === j ? 0 : ( i < j ? -1 : 1 ), comparison, i + ',' + j );
      }
    }
  } );
  
  test( 'TrailPointer render comparison', function() {
    var node = createTestNodeTree();
    
    equal( 0, new scenery.TrailPointer( node.getUniqueTrail(), true ).compareRender( new scenery.TrailPointer( node.getUniqueTrail(), true ) ), 'Same before pointer' );
    equal( 0, new scenery.TrailPointer( node.getUniqueTrail(), false ).compareRender( new scenery.TrailPointer( node.getUniqueTrail(), false ) ), 'Same after pointer' );
    equal( -1, new scenery.TrailPointer( node.getUniqueTrail(), true ).compareRender( new scenery.TrailPointer( node.getUniqueTrail(), false ) ), 'Same node before/after root' );
    equal( -1, new scenery.TrailPointer( node.children[0].getUniqueTrail(), true ).compareRender( new scenery.TrailPointer( node.children[0].getUniqueTrail(), false ) ), 'Same node before/after nonroot' );
    equal( 0, new scenery.TrailPointer( node.children[0].children[1].children[0].getUniqueTrail(), false ).compareRender( new scenery.TrailPointer( node.children[0].children[2].getUniqueTrail(), true ) ), 'Equivalence of before/after' );
    
    // all pointers in the render order
    var pointers = [
      new scenery.TrailPointer( node.getUniqueTrail(), true ),
      new scenery.TrailPointer( node.getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[0].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[0].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[1].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[1].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[1].children[0].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[1].children[0].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[2].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[2].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[3].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[3].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[3].children[0].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[3].children[0].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[3].children[0].children[0].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[3].children[0].children[0].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[3].children[1].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[3].children[1].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[4].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[4].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[1].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[1].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[2].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[2].getUniqueTrail(), false )
    ];
    
    // compare the pointers. different ones can be equal if they represent the same place, so we only check if they compare differently
    for ( var i = 0; i < pointers.length; i++ ) {
      for ( var j = i; j < pointers.length; j++ ) {
        var comparison = pointers[i].compareRender( pointers[j] );
        
        if ( comparison === -1 ) {
          ok( i < j, i + ',' + j );
        }
        if ( comparison === 1 ) {
          ok( i > j, i + ',' + j );
        }
      }
    }
  } );
  
  test( 'TrailPointer nested comparison and fowards/backwards', function() {
    var node = createTestNodeTree();
    
    // all pointers in the nested order
    var pointers = [
      new scenery.TrailPointer( node.getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[0].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[0].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[1].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[1].children[0].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[1].children[0].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[1].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[2].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[2].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[3].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[3].children[0].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[3].children[0].children[0].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[3].children[0].children[0].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[3].children[0].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[3].children[1].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[3].children[1].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[3].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].children[4].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[0].children[4].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[0].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[1].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[1].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[2].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[2].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.getUniqueTrail(), false )
    ];
    
    // exhaustively verify the ordering between each ordered pair
    for ( var i = 0; i < pointers.length; i++ ) {
      for ( var j = i; j < pointers.length; j++ ) {
        var comparison = pointers[i].compareNested( pointers[j] );
        
        // make sure that every pointer compares as expected (0 and they are equal, -1 and i < j)
        equal( comparison, i === j ? 0 : ( i < j ? -1 : 1 ), 'compareNested: ' + i + ',' + j );
      }
    }
    
    // verify forwards and backwards, as well as copy constructors
    for ( var i = 1; i < pointers.length; i++ ) {
      var a = pointers[i-1];
      var b = pointers[i];
      
      var forwardsCopy = a.copy();
      forwardsCopy.nestedForwards();
      equal( forwardsCopy.compareNested( b ), 0, 'forwardsPointerCheck ' + ( i - 1 ) + ' to ' + i );
      
      var backwardsCopy = b.copy();
      backwardsCopy.nestedBackwards();
      equal( backwardsCopy.compareNested( a ), 0, 'backwardsPointerCheck ' + i + ' to ' + ( i - 1 ) );
    }
    
    // exhaustively check depthFirstUntil inclusive
    for ( var i = 0; i < pointers.length; i++ ) {
      for ( var j = i + 1; j < pointers.length; j++ ) {
        // i < j guaranteed
        var contents = [];
        pointers[i].depthFirstUntil( pointers[j], function( pointer ) { contents.push( pointer.copy() ); }, false );
        equal( contents.length, j - i + 1, 'depthFirstUntil inclusive ' + i + ',' + j + ' count check' );
        
        // do an actual pointer to pointer comparison
        var isOk = true;
        for ( var k = 0; k < contents.length; k++ ) {
          var comparison = contents[k].compareNested( pointers[i+k] );
          if ( comparison !== 0 ) {
            equal( comparison, 0, 'depthFirstUntil inclusive ' + i + ',' + j + ',' + k + ' comparison check ' + contents[k].trail.indices.join() + ' - ' + pointers[i+k].trail.indices.join() );
            isOk = false;
          }
        }
        ok( isOk, 'depthFirstUntil inclusive ' + i + ',' + j + ' comparison check' );
      }
    }
    
    // exhaustively check depthFirstUntil exclusive
    for ( var i = 0; i < pointers.length; i++ ) {
      for ( var j = i + 1; j < pointers.length; j++ ) {
        // i < j guaranteed
        var contents = [];
        pointers[i].depthFirstUntil( pointers[j], function( pointer ) { contents.push( pointer.copy() ); }, true );
        equal( contents.length, j - i - 1, 'depthFirstUntil exclusive ' + i + ',' + j + ' count check' );
        
        // do an actual pointer to pointer comparison
        var isOk = true;
        for ( var k = 0; k < contents.length; k++ ) {
          var comparison = contents[k].compareNested( pointers[i+k+1] );
          if ( comparison !== 0 ) {
            equal( comparison, 0, 'depthFirstUntil exclusive ' + i + ',' + j + ',' + k + ' comparison check ' + contents[k].trail.indices.join() + ' - ' + pointers[i+k].trail.indices.join() );
            isOk = false;
          }
        }
        ok( isOk, 'depthFirstUntil exclusive ' + i + ',' + j + ' comparison check' );
      }
    }
  } );
  
  test( 'Text width measurement in canvas', function() {
    var canvas = document.createElement( 'canvas' );
    var context = phet.canvas.initCanvas( canvas );
    var metrics = context.measureText('Hello World');
    ok( metrics.width, 'metrics.width' );
  } );
  
  test( 'Sceneless node handling', function() {
    var a = new scenery.Path();
    var b = new scenery.Path();
    var c = new scenery.Path();
    
    a.setShape( scenery.Shape.rectangle( 0, 0, 20, 20 ) );
    c.setShape( scenery.Shape.rectangle( 10, 10, 30, 30 ) );
    
    a.addChild( b );
    b.addChild( c );
    
    a.validateBounds();
    
    a.removeChild( b );
    c.addChild( a );
    
    b.validateBounds();
    
    a.invalidatePaint();
    
    expect( 0 );
  } );
  
  test( 'Checking Layers and external canvas', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    scene.addChild( new scenery.Path( {
      shape: scenery.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#ff0000'
    } ) );
    
    var middleRect = new scenery.Path( {
      shape: scenery.Shape.rectangle( canvasWidth / 4, canvasHeight / 4, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#00ff00'
    } );
    middleRect.layerStrategy = scenery.SeparateLayerStrategy;
    
    scene.addChild( middleRect );
    
    scene.addChild( new scenery.Path( {
      shape: scenery.Shape.rectangle( canvasWidth / 2, canvasHeight / 2, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#0000ff'
    } ) );
    
    scene.updateScene();
    
    equal( scene.layers.length, 3, 'simple layer check' );
  } );
  
  test( 'Update vs Full Basic Clearing Check', function() {
    updateVsFullRender( [
      function( scene ) {
        scene.addChild( new scenery.Path( {
          shape: scenery.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ),
          fill: '#000000'
        } ) );
      }, function( scene ) {
        scene.children[0].translate( 20, 20 );
      }
    ] );
  } );
  
  test( 'Update vs Full Self-Bounds increase', function() {
    updateVsFullRender( [
      function( scene ) {
        var node = new scenery.Path();
        node.setShape( scenery.Shape.rectangle( 0, 0, canvasWidth / 3, canvasHeight / 3 ) );
        node.setFill( '#ff0000' );
        node.setStroke( '#000000' );
        scene.addChild( node );
      }, function( scene ) {
        scene.children[0].setShape( scenery.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ) );
      }
    ] );
  } );
  
  test( 'Update vs Full Stroke Repaint', function() {
    updateVsFullRender( [
      function( scene ) {
        // TODO: clearer way of specifying parameters
        var node = new scenery.Path();
        node.setShape( scenery.Shape.rectangle( 15, 15, canvasWidth / 2, canvasHeight / 2 ) );
        node.setFill( '#ff0000' );
        node.setStroke( '#000000' );
        node.setLineWidth( 10 );
        scene.addChild( node );
      }, function( scene ) {
        scene.children[0].translate( canvasWidth / 4, canvasHeight / 4 );
      }
    ] );
  } );
  
  test( 'Correct bounds on rectangle', function() {
    var rectBounds = scenery.canvasAccurateBounds( function( context ) { context.fillRect( 100, 100, 200, 200 ); } );
    ok( Math.abs( rectBounds.minX - 100 ) < 0.01, rectBounds.minX );
    ok( Math.abs( rectBounds.minY - 100 ) < 0.01, rectBounds.minY );
    ok( Math.abs( rectBounds.maxX - 300 ) < 0.01, rectBounds.maxX );
    ok( Math.abs( rectBounds.maxY - 300 ) < 0.01, rectBounds.maxY );
  } );
  
  test( 'Consistent and precise bounds range on Text', function() {
    var textBounds = scenery.canvasAccurateBounds( function( context ) { context.fillText( 'test string', 0, 0 ); } );
    ok( textBounds.isConsistent, textBounds.toString() );
    
    // precision of 0.001 (or lower given different parameters) is possible on non-Chome browsers (Firefox, IE9, Opera)
    ok( textBounds.precision < 0.15, 'precision: ' + textBounds.precision );
  } );
  
  test( 'ES5 Setter / Getter tests', function() {
    var node = new scenery.Path();
    var fill = '#abcdef';
    node.fill = fill;
    equal( node.fill, fill );
    equal( node.getFill(), fill );
    
    var otherNode = new scenery.Path( { fill: fill, shape: scenery.Shape.rectangle( 0, 0, 10, 10 ) } );
    
    equal( otherNode.fill, fill );
  } );
  
  test( 'Layer change stability', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    var root = scene;
    
    root.addChild( new scenery.Path( {
      shape: scenery.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#ff0000'
    } ) );
    
    var middleRect = new scenery.Path( {
      shape: scenery.Shape.rectangle( canvasWidth / 4, canvasHeight / 4, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#00ff00'
    } );
    
    
    root.addChild( middleRect );
    
    root.addChild( new scenery.Path( {
      shape: scenery.Shape.rectangle( canvasWidth / 2, canvasHeight / 2, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#0000ff'
    } ) );
    
    scene.updateScene();
    
    var snapshotA = snapshot( scene );
    
    middleRect.layerStrategy = scenery.SeparateLayerStrategy;
    scene.updateScene();
    
    var snapshotB = snapshot( scene );
    
    snapshotEquals( snapshotA, snapshotB, 0, 'Layer change stability' );
  } );
  
  test( 'Piccolo-like behavior', function() {
    var node = new scenery.Node();
    
    node.scaleBy( 2 );
    node.translate( 1, 3 );
    node.rotate( Math.PI / 2 );
    node.translate( -31, 21 );
    
    equalsApprox( node.getMatrix().m00(), 0 );
    equalsApprox( node.getMatrix().m01(), -2 );
    equalsApprox( node.getMatrix().m02(), -40 );
    equalsApprox( node.getMatrix().m10(), 2 );
    equalsApprox( node.getMatrix().m11(), 0 );
    equalsApprox( node.getMatrix().m12(), -56 );
    
    equalsApprox( node.x, -40 );
    equalsApprox( node.y, -56 );
    equalsApprox( node.rotation, Math.PI / 2 );
    
    node.translation = new phet.math.Vector2( -5, 7 );
    
    equalsApprox( node.getMatrix().m02(), -5 );
    equalsApprox( node.getMatrix().m12(), 7 );
    
    node.rotation = 1.2;
    
    equalsApprox( node.getMatrix().m01(), -1.864078171934453 );
    
    node.rotation = -0.7;
    
    equalsApprox( node.getMatrix().m10(), -1.288435374475382 );
    
    // console.log( node.getMatrix().toString() );
  } );
  
  test( 'Setting left/right of node', function() {
    var node = new scenery.Path( {
      shape: scenery.Shape.rectangle( -20, -20, 50, 50 ),
      scale: 2
    } );
    
    equalsApprox( node.left, -40 );
    equalsApprox( node.right, 60 );
    
    node.left = 10;
    equalsApprox( node.left, 10 );
    equalsApprox( node.right, 110 );
    
    node.right = 10;
    equalsApprox( node.left, -90 );
    equalsApprox( node.right, 10 );
    
    node.centerX = 5;
    equalsApprox( node.centerX, 5 );
    equalsApprox( node.left, -45 );
    equalsApprox( node.right, 55 );
  } );
  
  /*---------------------------------------------------------------------------*
  * Shapes
  *----------------------------------------------------------------------------*/    
  
  (function(){
    module( 'Shapes' );
    
    var Shape = scenery.Shape;
    
    function p( x, y ) { return new phet.math.Vector2( x, y ); }
  
    test( 'Verifying Line/Rect', function() {
      var lineWidth = 50;
      // /shapeToStroke, shapeToFill, strokeNodeSetup, message, debugFlag
      var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
      var fillShape = Shape.rectangle( 100, 100 - lineWidth / 2, 200, lineWidth );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineWidth( lineWidth ); }, QUnit.config.current.testName );
    } );
    
    test( 'Line Segment - butt', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 50;
      
      var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    test( 'Line Segment - square', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 50;
      styles.lineCap = 'square';
      
      var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    test( 'Line Join - Miter', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 30;
      styles.lineJoin = 'miter';
      
      var strokeShape = new Shape();
      strokeShape.moveTo( 70, 70 );
      strokeShape.lineTo( 140, 200 );
      strokeShape.lineTo( 210, 70 );
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    test( 'Line Join - Miter - Closed', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 30;
      styles.lineJoin = 'miter';
      
      var strokeShape = new Shape();
      strokeShape.moveTo( 70, 70 );
      strokeShape.lineTo( 140, 200 );
      strokeShape.lineTo( 210, 70 );
      strokeShape.close();
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    test( 'Line Join - Bevel - Closed', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 30;
      styles.lineJoin = 'bevel';
      
      var strokeShape = new Shape();
      strokeShape.moveTo( 70, 70 );
      strokeShape.lineTo( 140, 200 );
      strokeShape.lineTo( 210, 70 );
      strokeShape.close();
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    test( 'Rect', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 30;
      
      var strokeShape = Shape.rectangle( 40, 40, 150, 150 );
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    test( 'Manual Rect', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 30;
      
      var strokeShape = new Shape();
      strokeShape.moveTo( 40, 40 );
      strokeShape.lineTo( 190, 40 );
      strokeShape.lineTo( 190, 190 );
      strokeShape.lineTo( 40, 190 );
      strokeShape.lineTo( 40, 40 );
      strokeShape.close();
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    test( 'Hex', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 30;
      
      var strokeShape = Shape.regularPolygon( 6, 100 ).transformed( phet.math.Matrix3.translation( 130, 130 ) );
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    test( 'Overlap', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 30;
      
      var strokeShape = new Shape();
      strokeShape.moveTo( 40, 40 );
      strokeShape.lineTo( 200, 200 );
      strokeShape.lineTo( 40, 200 );
      strokeShape.lineTo( 200, 40 );
      strokeShape.lineTo( 100, 0 );
      strokeShape.close();
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    var miterMagnitude = 160;
    var miterAnglesInDegrees = [5, 8, 10, 11.5, 13, 20, 24, 30, 45];
    
    _.each( miterAnglesInDegrees, function( miterAngle ) {
      var miterAngleRadians = miterAngle * Math.PI / 180;
      test( 'Miter limit angle (degrees): ' + miterAngle + ' would change at ' + 1 / Math.sin( miterAngleRadians / 2 ), function() {
        var styles = new Shape.LineStyles();
        styles.lineWidth = 30;
        
        var strokeShape = new Shape();
        var point = new phet.math.Vector2( 40, 100 );
        strokeShape.moveTo( point );
        point = point.plus( phet.math.Vector2.X_UNIT.times( miterMagnitude ) );
        strokeShape.lineTo( point );
        point = point.plus( phet.math.Vector2.createPolar( miterMagnitude, miterAngleRadians ).negated() );
        strokeShape.lineTo( point );
        var fillShape = strokeShape.getStrokedShape( styles );
        
        strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
      } );
    } );
    
    test( 'Overlapping rectangles', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 30;
      
      var strokeShape = new Shape();
      strokeShape.rect( 40, 40, 100, 100 );
      strokeShape.rect( 50, 50, 100, 100 );
      strokeShape.rect( 80, 80, 100, 100 );
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    test( 'Line segment winding', function() {
      var line = new Shape.Segment.Line( p( 0, 0 ), p( 2, 2 ) );
      
      equal( line.windingIntersection( new phet.math.Ray2( p( 0, 1 ), p( 1, 0 ) ) ), 1 );
      equal( line.windingIntersection( new phet.math.Ray2( p( 0, 5 ), p( 1, 0 ) ) ), 0 );
      equal( line.windingIntersection( new phet.math.Ray2( p( 1, 0 ), p( 0, 1 ) ) ), -1 );
      equal( line.windingIntersection( new phet.math.Ray2( p( 0, 0 ), p( 1, 1 ).normalized() ) ), 0 );
      equal( line.windingIntersection( new phet.math.Ray2( p( 0, 1 ), p( 1, 1 ).normalized() ) ), 0 );
    } );
    
    test( 'Rectangle hit testing', function() {
      var shape = Shape.rectangle( 0, 0, 1, 1 );
      
      equal( shape.containsPoint( p( 0.2, 0.3 ) ), true, '0.2, 0.3' );
      equal( shape.containsPoint( p( 0.5, 0.5 ) ), true, '0.5, 0.5' );
      equal( shape.containsPoint( p( 1.5, 0.5 ) ), false, '1.5, 0.5' );
      equal( shape.containsPoint( p( -0.5, 0.5 ) ), false, '-0.5, 0.5' );
    } );
    
    test( 'Bezier Offset', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 30;
      
      var strokeShape = new Shape();
      strokeShape.moveTo( 40, 40 );
      strokeShape.quadraticCurveTo( 100, 200, 160, 40 );
      // strokeShape.close();
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
    
    test( 'Bezier Edge Case (literally)', function() {
      var styles = new Shape.LineStyles();
      styles.lineWidth = 30;
      
      var strokeShape = new Shape();
      strokeShape.moveTo( 40, 40 );
      strokeShape.quadraticCurveTo( 200, 200, 200, 180 );
      // strokeShape.close();
      var fillShape = strokeShape.getStrokedShape( styles );
      
      strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
    } );
  })();
  
  /*---------------------------------------------------------------------------*
  * Text
  *----------------------------------------------------------------------------*/    
  
  module( 'Text' );
  
  test( 'Canvas Accurate Text Bounds', function() {
    testTextBounds( scenery.canvasTextBoundsAccurate, {
      font: '10px sans-serif',
      textAlign: 'left', // left is not the default, 'start' is
      textBaseline: 'alphabetic',
      direction: 'ltr'
    }, QUnit.config.current.testName );
  } );
  
  /*---------------------------------------------------------------------------*
  * DOM
  *----------------------------------------------------------------------------*/    
  
  // module( 'DOM Layers' );
  
  // test( 'DOM Test', function() {
  //   updateVsFullRender( [
  //     function( scene ) {
  //       var node = new scenery.Path();
  //       node.setShape( scenery.Shape.rectangle( 0, 0, canvasWidth / 3, canvasHeight / 3 ) );
  //       node.setFill( '#ff0000' );
  //       node.setStroke( '#000000' );
  //       scene.addChild( node );
        
  //       var domNode = new scenery.Node();
  //       node.addChild( domNode );
  //     }
  //   ] );
  // } );
  
  /*---------------------------------------------------------------------------*
  * Miscellaneous HTML / JS
  *----------------------------------------------------------------------------*/    
  
  module( 'Miscellaneous' );
  
  test( 'ES5 Object.defineProperty get/set', function() {
    var ob = { _key: 5 };
    Object.defineProperty( ob, 'key', {
      enumerable: true,
      configurable: true,
      get: function() { return this._key; },
      set: function( val ) { this._key = val; }
    } );
    ob.key += 1;
    equal( ob._key, 6, 'incremented object value' );
  } );
  
  /*---------------------------------------------------------------------------*
  * WebGL
  *----------------------------------------------------------------------------*/    
  
  module( 'WebGL tests' );
  
  test( 'Canvas WebGL Context and Features', function() {
    var canvas = document.createElement( 'canvas' );
    var context = phet.webgl.initWebGL( canvas );
    ok( context, 'context' );
  } );
  
  /*---------------------------------------------------------------------------*
  * Canvas V5 (NEW)
  *----------------------------------------------------------------------------*/    
  
  if ( includeBleedingEdgeCanvasTests ) {
    // v5 canvas additions
    module( 'Bleeding Edge Canvas Support' );
    
    test( 'Canvas 2D v5 Features', function() {
      var canvas = document.createElement( 'canvas' );
      var context = phet.canvas.initCanvas( canvas );
      
      var neededMethods = [
        'addHitRegion',
        'ellipse',
        'resetClip',
        'resetTransform'
      ];
      _.each( neededMethods, function( method ) {
        ok( context[method] !== undefined, 'context.' + method );
      } );
    } );
    
    test( 'Path object support', function() {
      var path = new Path();
    } );
       
    test( 'Text width measurement in canvas', function() {
      var canvas = document.createElement( 'canvas' );
      var context = phet.canvas.initCanvas( canvas );
      var metrics = context.measureText('Hello World');
      _.each( [ 'actualBoundingBoxLeft', 'actualBoundingBoxRight', 'actualBoundingBoxAscent', 'actualBoundingBoxDescent' ], function( method ) {
        ok( metrics[method] !== undefined, 'metrics.' + method );
      } );
    } );
  }
  
})();



