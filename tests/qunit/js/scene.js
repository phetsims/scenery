(function() {
  'use strict';

  module( 'Scenery: Scene' );

  test( 'Dirty bounds propagation test', function() {
    var node = createTestNodeTree();

    node.validateBounds();

    ok( !node._childBoundsDirty );

    node.children[ 0 ].children[ 3 ].children[ 0 ].invalidateBounds();

    ok( node._childBoundsDirty );
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
      ok( context[ method ] !== undefined, 'context.' + method );
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
        var comparison = trails[ i ].compare( trails[ j ] );

        // make sure that every trail compares as expected (0 and they are equal, -1 and i < j)
        equal( i === j ? 0 : ( i < j ? -1 : 1 ), comparison, i + ',' + j );
      }
    }
  } );

  test( 'Trail eachTrailBetween', function() {
    var node = createTestNodeTree();

    // get a list of all trails in render order
    var trails = [];
    var currentTrail = new scenery.Trail( node ); // start at the first node

    while ( currentTrail ) {
      trails.push( currentTrail );
      currentTrail = currentTrail.next();
    }

    equal( 13, trails.length, 'Trails: ' + _.map( trails, function( trail ) { return trail.toString(); } ).join( '\n' ) );

    for ( var i = 0; i < trails.length; i++ ) {
      for ( var j = i; j < trails.length; j++ ) {
        var inclusiveList = [];
        scenery.Trail.eachTrailBetween( trails[ i ], trails[ j ], function( trail ) {
          inclusiveList.push( trail.copy() );
        }, false, node );
        var trailString = i + ',' + j + ' ' + trails[ i ].toString() + ' to ' + trails[ j ].toString()
        ok( inclusiveList[ 0 ].equals( trails[ i ] ), 'inclusive start on ' + trailString + ' is ' + inclusiveList[ 0 ].toString() );
        ok( inclusiveList[ inclusiveList.length - 1 ].equals( trails[ j ] ), 'inclusive end on ' + trailString + 'is ' + inclusiveList[ inclusiveList.length - 1 ].toString() );
        equal( inclusiveList.length, j - i + 1, 'inclusive length on ' + trailString + ' is ' + inclusiveList.length + ', ' + _.map( inclusiveList, function( trail ) { return trail.toString(); } ).join( '\n' ) );

        if ( i < j ) {
          var exclusiveList = [];
          scenery.Trail.eachTrailBetween( trails[ i ], trails[ j ], function( trail ) {
            exclusiveList.push( trail.copy() );
          }, true, node );
          equal( exclusiveList.length, j - i - 1, 'exclusive length on ' + i + ',' + j );
        }
      }
    }
  } );

  test( 'depthFirstUntil depthFirstUntil with subtree skipping', function() {
    var node = createTestNodeTree();
    node.children[ 0 ].children[ 2 ].visible = false;
    node.children[ 0 ].children[ 3 ].visible = false;
    new scenery.TrailPointer( new scenery.Trail( node ), true ).depthFirstUntil( new scenery.TrailPointer( new scenery.Trail( node ), false ), function( pointer ) {
      if ( !pointer.trail.lastNode().isVisible() ) {
        // should skip
        return true;
      }
      ok( pointer.trail.isVisible(), 'Trail visibility for ' + pointer.trail.toString() );
    }, false );
  } );

  test( 'Trail eachTrailUnder with subtree skipping', function() {
    var node = createTestNodeTree();
    node.children[ 0 ].children[ 2 ].visible = false;
    node.children[ 0 ].children[ 3 ].visible = false;
    new scenery.Trail( node ).eachTrailUnder( function( trail ) {
      if ( !trail.lastNode().isVisible() ) {
        // should skip
        return true;
      }
      ok( trail.isVisible(), 'Trail visibility for ' + trail.toString() );
    } );
  } );

  test( 'TrailPointer render comparison', function() {
    var node = createTestNodeTree();

    equal( 0, new scenery.TrailPointer( node.getUniqueTrail(), true ).compareRender( new scenery.TrailPointer( node.getUniqueTrail(), true ) ), 'Same before pointer' );
    equal( 0, new scenery.TrailPointer( node.getUniqueTrail(), false ).compareRender( new scenery.TrailPointer( node.getUniqueTrail(), false ) ), 'Same after pointer' );
    equal( -1, new scenery.TrailPointer( node.getUniqueTrail(), true ).compareRender( new scenery.TrailPointer( node.getUniqueTrail(), false ) ), 'Same node before/after root' );
    equal( -1, new scenery.TrailPointer( node.children[ 0 ].getUniqueTrail(), true ).compareRender( new scenery.TrailPointer( node.children[ 0 ].getUniqueTrail(), false ) ), 'Same node before/after nonroot' );
    equal( 0, new scenery.TrailPointer( node.children[ 0 ].children[ 1 ].children[ 0 ].getUniqueTrail(), false ).compareRender( new scenery.TrailPointer( node.children[ 0 ].children[ 2 ].getUniqueTrail(), true ) ), 'Equivalence of before/after' );

    // all pointers in the render order
    var pointers = [
      new scenery.TrailPointer( node.getUniqueTrail(), true ),
      new scenery.TrailPointer( node.getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 0 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 0 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 1 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 1 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 1 ].children[ 0 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 1 ].children[ 0 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 2 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 2 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].children[ 0 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].children[ 0 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 1 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 1 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 4 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 4 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 1 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 1 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 2 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 2 ].getUniqueTrail(), false )
    ];

    // compare the pointers. different ones can be equal if they represent the same place, so we only check if they compare differently
    for ( var i = 0; i < pointers.length; i++ ) {
      for ( var j = i; j < pointers.length; j++ ) {
        var comparison = pointers[ i ].compareRender( pointers[ j ] );

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
      new scenery.TrailPointer( node.children[ 0 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 0 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 0 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 1 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 1 ].children[ 0 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 1 ].children[ 0 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 1 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 2 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 2 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].children[ 0 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].children[ 0 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 1 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].children[ 1 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 3 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 4 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 0 ].children[ 4 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 0 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 1 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 1 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.children[ 2 ].getUniqueTrail(), true ),
      new scenery.TrailPointer( node.children[ 2 ].getUniqueTrail(), false ),
      new scenery.TrailPointer( node.getUniqueTrail(), false )
    ];

    // exhaustively verify the ordering between each ordered pair
    for ( var i = 0; i < pointers.length; i++ ) {
      for ( var j = i; j < pointers.length; j++ ) {
        var comparison = pointers[ i ].compareNested( pointers[ j ] );

        // make sure that every pointer compares as expected (0 and they are equal, -1 and i < j)
        equal( comparison, i === j ? 0 : ( i < j ? -1 : 1 ), 'compareNested: ' + i + ',' + j );
      }
    }

    // verify forwards and backwards, as well as copy constructors
    for ( var i = 1; i < pointers.length; i++ ) {
      var a = pointers[ i - 1 ];
      var b = pointers[ i ];

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
        pointers[ i ].depthFirstUntil( pointers[ j ], function( pointer ) { contents.push( pointer.copy() ); }, false );
        equal( contents.length, j - i + 1, 'depthFirstUntil inclusive ' + i + ',' + j + ' count check' );

        // do an actual pointer to pointer comparison
        var isOk = true;
        for ( var k = 0; k < contents.length; k++ ) {
          var comparison = contents[ k ].compareNested( pointers[ i + k ] );
          if ( comparison !== 0 ) {
            equal( comparison, 0, 'depthFirstUntil inclusive ' + i + ',' + j + ',' + k + ' comparison check ' + contents[ k ].trail.indices.join() + ' - ' + pointers[ i + k ].trail.indices.join() );
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
        pointers[ i ].depthFirstUntil( pointers[ j ], function( pointer ) { contents.push( pointer.copy() ); }, true );
        equal( contents.length, j - i - 1, 'depthFirstUntil exclusive ' + i + ',' + j + ' count check' );

        // do an actual pointer to pointer comparison
        var isOk = true;
        for ( var k = 0; k < contents.length; k++ ) {
          var comparison = contents[ k ].compareNested( pointers[ i + k + 1 ] );
          if ( comparison !== 0 ) {
            equal( comparison, 0, 'depthFirstUntil exclusive ' + i + ',' + j + ',' + k + ' comparison check ' + contents[ k ].trail.indices.join() + ' - ' + pointers[ i + k ].trail.indices.join() );
            isOk = false;
          }
        }
        ok( isOk, 'depthFirstUntil exclusive ' + i + ',' + j + ' comparison check' );
      }
    }
  } );

  // test( 'TrailInterval', function() {
  //   var node = createTestNodeTree();
  //   var i, j;

  //   // a subset of trails to test on
  //   var trails = [
  //     null,
  //     node.children[0].getUniqueTrail(),
  //     node.children[0].children[1].getUniqueTrail(), // commented out since it quickly creates many tests to include
  //     node.children[0].children[3].children[0].getUniqueTrail(),
  //     node.children[1].getUniqueTrail(),
  //     null
  //   ];

  //   // get a list of all trails
  //   var allTrails = [];
  //   var t = node.getUniqueTrail();
  //   while ( t ) {
  //     allTrails.push( t );
  //     t = t.next();
  //   }

  //   // get a list of all intervals using our 'trails' array
  //   var intervals = [];

  //   for ( i = 0; i < trails.length; i++ ) {
  //     // only create proper intervals where i < j, since we specified them in order
  //     for ( j = i + 1; j < trails.length; j++ ) {
  //       var interval = new scenery.TrailInterval( trails[i], trails[j] );
  //       intervals.push( interval );

  //       // tag the interval, so we can do additional verification later
  //       interval.i = i;
  //       interval.j = j;
  //     }
  //   }

  //   // check every combination of intervals
  //   for ( i = 0; i < intervals.length; i++ ) {
  //     var a = intervals[i];
  //     for ( j = 0; j < intervals.length; j++ ) {
  //       var b = intervals[j];

  //       var union = a.union( b );
  //       if ( a.exclusiveUnionable( b ) ) {
  //         _.each( allTrails, function( trail ) {
  //           if ( trail ) {
  //             var msg = 'union check of trail ' + trail.toString() + ' with ' + a.toString() + ' and ' + b.toString() + ' with union ' + union.toString();
  //             equal( a.exclusiveContains( trail ) || b.exclusiveContains( trail ), union.exclusiveContains( trail ), msg );
  //           }
  //         } );
  //       } else {
  //         var wouldBeBadUnion = false;
  //         var containsAnything = false;
  //         _.each( allTrails, function( trail ) {
  //           if ( trail ) {
  //             if ( union.exclusiveContains( trail ) ) {
  //               containsAnything = true;
  //               if ( !a.exclusiveContains( trail ) && !b.exclusiveContains( trail ) ) {
  //                 wouldBeBadUnion = true;
  //               }
  //             }
  //           }
  //         } );
  //         ok( containsAnything && wouldBeBadUnion, 'Not a bad union?: ' + a.toString() + ' and ' + b.toString() + ' with union ' + union.toString() );
  //       }
  //     }
  //   }
  // } );

  test( 'Text width measurement in canvas', function() {
    var canvas = document.createElement( 'canvas' );
    var context = canvas.getContext( '2d' );
    var metrics = context.measureText( 'Hello World' );
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

    expect( 0 );
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

    var otherNode = new scenery.Path( kite.Shape.rectangle( 0, 0, 10, 10 ), { fill: fill } );

    equal( otherNode.fill, fill );
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
    var node = new scenery.Path( kite.Shape.rectangle( -20, -20, 50, 50 ), {
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
    var scene = new scenery.Node();

    var node = new scenery.Path( new kite.Shape() );

    scene.addChild( node );
    expect( 0 );
  } );

  test( 'Path with null shape', function() {
    var scene = new scenery.Node();

    var node = new scenery.Path( null );

    scene.addChild( node );
    expect( 0 );
  } );

  test( 'Display resize event', function() {
    var scene = new scenery.Node();
    var display = new scenery.Display( scene );

    var width, height, count = 0;

    display.on( 'displaySize', function( size ) {
      width = size.width;
      height = size.height;
      count++;
    } );

    display.setWidthHeight( 712, 217 );

    equal( width, 712, 'Scene resize width' );
    equal( height, 217, 'Scene resize height' );
    equal( count, 1, 'Scene resize count' );
  } );

  test( 'Bounds events', function() {
    var node = new scenery.Node();
    node.y = 10;

    var rect = new scenery.Rectangle( 0, 0, 100, 50, { fill: '#f00' } );
    rect.x = 10; // a transform, so we can verify everything is handled correctly
    node.addChild( rect );

    node.validateBounds();

    var epsilon = 0.0000001;

    node.addEventListener( 'childBounds', function() {
      ok( node.childBounds.equalsEpsilon( new dot.Bounds2( 10, 0, 110, 30 ), epsilon ), 'Parent child bounds check: ' + node.childBounds.toString() );
    } );

    node.addEventListener( 'bounds', function() {
      ok( node.bounds.equalsEpsilon( new dot.Bounds2( 10, 10, 110, 40 ), epsilon ), 'Parent bounds check: ' + node.bounds.toString() );
    } );

    node.addEventListener( 'selfBounds', function() {
      ok( false, 'Self bounds should not change for parent node' );
    } );

    rect.addEventListener( 'selfBounds', function() {
      ok( rect.selfBounds.equalsEpsilon( new dot.Bounds2( 0, 0, 100, 30 ), epsilon ), 'Self bounds check: ' + rect.selfBounds.toString() );
    } );

    rect.addEventListener( 'bounds', function() {
      ok( rect.bounds.equalsEpsilon( new dot.Bounds2( 10, 0, 110, 30 ), epsilon ), 'Bounds check: ' + rect.bounds.toString() );
    } );

    rect.addEventListener( 'childBounds', function() {
      ok( false, 'Child bounds should not change for leaf node' );
    } );

    rect.rectHeight = 30;
    node.validateBounds();

    // this may change if for some reason we end up calling more events in the future
    expect( 4 );
  } );

  test( 'Using a color instance', function() {
    var scene = new scenery.Node();

    var rect = new scenery.Rectangle( 0, 0, 100, 50 );
    ok( rect.fill === null, 'Always starts with a null fill' );
    scene.addChild( rect );
    var color = new scenery.Color( 255, 0, 0 );
    rect.fill = color;
    color.setRGBA( 0, 255, 0, 1 );
  } );

  test( 'Bounds and Visible Bounds', function() {
    var node = new scenery.Node();
    var rect = new scenery.Rectangle( 0, 0, 100, 50 );
    node.addChild( rect );

    ok( node.visibleBounds.equals( new dot.Bounds2( 0, 0, 100, 50 ) ), 'Visible Bounds Visible' );
    ok( node.bounds.equals( new dot.Bounds2( 0, 0, 100, 50 ) ), 'Complete Bounds Visible' );

    rect.visible = false;

    ok( node.visibleBounds.equals( dot.Bounds2.NOTHING ), 'Visible Bounds Invisible' );
    ok( node.bounds.equals( new dot.Bounds2( 0, 0, 100, 50 ) ), 'Complete Bounds Invisible' );
  } );

  test( 'localBounds override', function() {
    var node = new scenery.Node( { y: 5 } );
    var rect = new scenery.Rectangle( 0, 0, 100, 50 );
    node.addChild( rect );

    rect.localBounds = new dot.Bounds2( 0, 0, 50, 50 );
    ok( node.localBounds.equals( new dot.Bounds2( 0, 0, 50, 50 ) ), 'localBounds override on self' );
    ok( node.bounds.equals( new dot.Bounds2( 0, 5, 50, 55 ) ), 'localBounds override on self' );

    rect.localBounds = new dot.Bounds2( 0, 0, 50, 100 );
    ok( node.bounds.equals( new dot.Bounds2( 0, 5, 50, 105 ) ), 'localBounds override 2nd on self' );

    // reset local bounds (have them computed again)
    rect.localBounds = null;
    ok( node.bounds.equals( new dot.Bounds2( 0, 5, 100, 55 ) ), 'localBounds override reset on self' );

    node.localBounds = new dot.Bounds2( 0, 0, 50, 200 );
    ok( node.localBounds.equals( new dot.Bounds2( 0, 0, 50, 200 ) ), 'localBounds override on parent' );
    ok( node.bounds.equals( new dot.Bounds2( 0, 5, 50, 205 ) ), 'localBounds override on parent' );
  } );

  test( 'fillColor/strokeColor', function() {
    var rect = new scenery.Rectangle( 0, 0, 100, 50, { fill: 'red', stroke: 'rgba(0,255,0,0.5)' } );
    equal( rect.fillColor.red, 255, 'Fill red' );
    equal( rect.fillColor.green, 0, 'Fill green' );
    equal( rect.fillColor.blue, 0, 'Fill blue' );
    equal( rect.fillColor.alpha, 1, 'Fill alpha' );
    equal( rect.strokeColor.red, 0, 'Stroke red' );
    equal( rect.strokeColor.green, 255, 'Stroke green' );
    equal( rect.strokeColor.blue, 0, 'Stroke blue' );
    equal( rect.strokeColor.alpha, 0.5, 'Stroke alpha' );
    rect.fill = rect.stroke;
    equal( rect.fillColor.red, 0, 'Fill red after change' );
    equal( rect.fillColor.green, 255, 'Fill green after change' );
    equal( rect.fillColor.blue, 0, 'Fill blue after change' );
    equal( rect.fillColor.alpha, 0.5, 'Fill alpha after change' );
    rect.stroke = '#ff0';
    equal( rect.strokeColor.red, 255, 'Stroke red after change' );
    equal( rect.strokeColor.green, 255, 'Stroke green after change' );
    equal( rect.strokeColor.blue, 0, 'Stroke blue after change' );
    equal( rect.strokeColor.alpha, 1, 'Stroke alpha after change' );
  } );
})();
