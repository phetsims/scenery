
(function(){
  'use strict';
  
  module( 'Scenery: Scene' );
  
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
    var context = canvas.getContext( '2d' );
    
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
  
  test( 'TrailInterval', function() {
    var node = createTestNodeTree();
    var i, j;
    
    // a subset of trails to test on
    var trails = [
      null,
      node.getUniqueTrail(),
      // node.children[0].children[1].getUniqueTrail(), // commented out since it quickly creates many tests to include
      node.children[0].children[3].children[0].getUniqueTrail(),
      node.children[1].getUniqueTrail(),
      null
    ];
    
    var intervals = [];
    
    for ( i = 0; i < trails.length; i++ ) {
      // only create proper intervals where i < j, since we specified them in order
      for ( j = i + 1; j < trails.length; j++ ) {
        intervals.push( new scenery.TrailInterval( trails[i], trails[j] ) );
      }
    }
    
    // check every combination of intervals
    for ( i = 0; i < intervals.length; i++ ) {
      var a = intervals[i];
      for ( j = 0; j < intervals.length; j++ ) {
        var b = intervals[j];
        
        if ( a.exclusiveUnionable( b ) ) {
          var union = a.union( b );
          _.each( trails, function( trail ) {
            if ( trail ) {
              var msg = 'union check of trail ' + trail.toString() + ' with ' + a.toString() + ' and ' + b.toString() + ' with union ' + union.toString();
              equal( a.exclusiveContains( trail ) || b.exclusiveContains( trail ), union.exclusiveContains( trail ), msg );
            }
          } );
        }
      }
    }
  } );
  
  test( 'Text width measurement in canvas', function() {
    var canvas = document.createElement( 'canvas' );
    var context = canvas.getContext( '2d' );
    var metrics = context.measureText('Hello World');
    ok( metrics.width, 'metrics.width' );
  } );
  
  test( 'Sceneless node handling', function() {
    var a = new scenery.Path();
    var b = new scenery.Path();
    var c = new scenery.Path();
    
    a.setShape( kite.Shape.rectangle( 0, 0, 20, 20 ) );
    c.setShape( kite.Shape.rectangle( 10, 10, 30, 30 ) );
    
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
      shape: kite.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#ff0000'
    } ) );
    
    var middleRect = new scenery.Path( {
      shape: kite.Shape.rectangle( canvasWidth / 4, canvasHeight / 4, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#00ff00'
    } );
    middleRect.layerSplit = true;
    
    scene.addChild( middleRect );
    
    scene.addChild( new scenery.Path( {
      shape: kite.Shape.rectangle( canvasWidth / 2, canvasHeight / 2, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#0000ff'
    } ) );
    
    scene.updateScene();
    
    equal( scene.layers.length, 3, 'simple layer check' );
  } );
  
  test( 'Update vs Full Basic Clearing Check', function() {
    updateVsFullRender( [
      function( scene ) {
        scene.addChild( new scenery.Path( {
          shape: kite.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ),
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
        node.setShape( kite.Shape.rectangle( 0, 0, canvasWidth / 3, canvasHeight / 3 ) );
        node.setFill( '#ff0000' );
        node.setStroke( '#000000' );
        scene.addChild( node );
      }, function( scene ) {
        scene.children[0].setShape( kite.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ) );
      }
    ] );
  } );
  
  test( 'Update vs Full Stroke Repaint', function() {
    updateVsFullRender( [
      function( scene ) {
        // TODO: clearer way of specifying parameters
        var node = new scenery.Path();
        node.setShape( kite.Shape.rectangle( 15, 15, canvasWidth / 2, canvasHeight / 2 ) );
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
    var rectBounds = scenery.Util.canvasAccurateBounds( function( context ) { context.fillRect( 100, 100, 200, 200 ); } );
    ok( Math.abs( rectBounds.minX - 100 ) < 0.01, rectBounds.minX );
    ok( Math.abs( rectBounds.minY - 100 ) < 0.01, rectBounds.minY );
    ok( Math.abs( rectBounds.maxX - 300 ) < 0.01, rectBounds.maxX );
    ok( Math.abs( rectBounds.maxY - 300 ) < 0.01, rectBounds.maxY );
  } );
  
  test( 'Consistent and precise bounds range on Text', function() {
    var textBounds = scenery.Util.canvasAccurateBounds( function( context ) { context.fillText( 'test string', 0, 0 ); } );
    ok( textBounds.isConsistent, textBounds.toString() );
    
    // precision of 0.001 (or lower given different parameters) is possible on non-Chome browsers (Firefox, IE9, Opera)
    ok( textBounds.precision < 0.15, 'precision: ' + textBounds.precision );
  } );
  
  test( 'Consistent and precise bounds range on Text', function() {
    var text = new scenery.Text( '0\u0489' );
    var textBounds = text.accurateCanvasBounds();
    ok( textBounds.isConsistent, textBounds.toString() );
    
    // precision of 0.001 (or lower given different parameters) is possible on non-Chome browsers (Firefox, IE9, Opera)
    ok( textBounds.precision < 1, 'precision: ' + textBounds.precision );
  } );
  
  test( 'ES5 Setter / Getter tests', function() {
    var node = new scenery.Path();
    var fill = '#abcdef';
    node.fill = fill;
    equal( node.fill, fill );
    equal( node.getFill(), fill );
    
    var otherNode = new scenery.Path( { fill: fill, shape: kite.Shape.rectangle( 0, 0, 10, 10 ) } );
    
    equal( otherNode.fill, fill );
  } );
  
  test( 'Layer change stability', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    var root = scene;
    
    root.addChild( new scenery.Path( {
      shape: kite.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#ff0000'
    } ) );
    
    var middleRect = new scenery.Path( {
      shape: kite.Shape.rectangle( canvasWidth / 4, canvasHeight / 4, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#00ff00'
    } );
    
    
    root.addChild( middleRect );
    
    root.addChild( new scenery.Path( {
      shape: kite.Shape.rectangle( canvasWidth / 2, canvasHeight / 2, canvasWidth / 2, canvasHeight / 2 ),
      fill: '#0000ff'
    } ) );
    
    scene.updateScene();
    
    var snapshotA = snapshot( scene );
    
    middleRect.layerSplit = true;
    scene.updateScene();
    
    var snapshotB = snapshot( scene );
    
    snapshotEquals( snapshotA, snapshotB, 0, 'Layer change stability' );
  } );
  
  test( 'Piccolo-like behavior', function() {
    var node = new scenery.Node();
    
    node.scale( 2 );
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
    
    node.translation = new dot.Vector2( -5, 7 );
    
    equalsApprox( node.getMatrix().m02(), -5 );
    equalsApprox( node.getMatrix().m12(), 7 );
    
    node.rotation = 1.2;
    
    equalsApprox( node.getMatrix().m01(), -1.864078171934453 );
    
    node.rotation = -0.7;
    
    equalsApprox( node.getMatrix().m10(), -1.288435374475382 );
  } );
  
  test( 'Setting left/right of node', function() {
    var node = new scenery.Path( {
      shape: kite.Shape.rectangle( -20, -20, 50, 50 ),
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
  
  test( 'Path with empty shape', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    var node = new scenery.Path( {
      shape: new kite.Shape()
    } );
    
    scene.addChild( node );
    scene.updateScene();
    expect( 0 );
  } );
  
  test( 'Path with null shape', function() {
    var scene = new scenery.Scene( $( '#main' ) );
    
    var node = new scenery.Path( {
      shape: null
    } );
    
    scene.addChild( node );
    scene.updateScene();
    expect( 0 );
  } );
})();
