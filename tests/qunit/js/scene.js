// Copyright 2002-2014, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: Scene' );

  /* eslint-disable no-undef */

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
        var trailString = i + ',' + j + ' ' + trails[ i ].toString() + ' to ' + trails[ j ].toString();
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
    for ( i = 1; i < pointers.length; i++ ) {
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
    for ( i = 0; i < pointers.length; i++ ) {
      for ( j = i + 1; j < pointers.length; j++ ) {
        // i < j guaranteed
        var contents = [];
        pointers[ i ].depthFirstUntil( pointers[ j ], function( pointer ) { contents.push( pointer.copy() ); }, false );
        equal( contents.length, j - i + 1, 'depthFirstUntil inclusive ' + i + ',' + j + ' count check' );

        // do an actual pointer to pointer comparison
        var isOk = true;
        for ( var k = 0; k < contents.length; k++ ) {
          comparison = contents[ k ].compareNested( pointers[ i + k ] );
          if ( comparison !== 0 ) {
            equal( comparison, 0, 'depthFirstUntil inclusive ' + i + ',' + j + ',' + k + ' comparison check ' + contents[ k ].trail.indices.join() + ' - ' + pointers[ i + k ].trail.indices.join() );
            isOk = false;
          }
        }
        ok( isOk, 'depthFirstUntil inclusive ' + i + ',' + j + ' comparison check' );
      }
    }

    // exhaustively check depthFirstUntil exclusive
    for ( i = 0; i < pointers.length; i++ ) {
      for ( j = i + 1; j < pointers.length; j++ ) {
        // i < j guaranteed
        contents = [];
        pointers[ i ].depthFirstUntil( pointers[ j ], function( pointer ) { contents.push( pointer.copy() ); }, true );
        equal( contents.length, j - i - 1, 'depthFirstUntil exclusive ' + i + ',' + j + ' count check' );

        // do an actual pointer to pointer comparison
        isOk = true;
        for ( k = 0; k < contents.length; k++ ) {
          comparison = contents[ k ].compareNested( pointers[ i + k + 1 ] );
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
    var a = new scenery.Path( null );
    var b = new scenery.Path( null );
    var c = new scenery.Path( null );

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
    var textBounds = scenery.TextBounds.accurateCanvasBounds( text );
    ok( textBounds.isConsistent, textBounds.toString() );

    // precision of 0.001 (or lower given different parameters) is possible on non-Chome browsers (Firefox, IE9, Opera)
    ok( textBounds.precision < 1, 'precision: ' + textBounds.precision );
  } );

  test( 'ES5 Setter / Getter tests', function() {
    var node = new scenery.Path( null );
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

    var width;
    var height;
    var count = 0;

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

    node.on( 'childBounds', function() {
      ok( node.childBounds.equalsEpsilon( new dot.Bounds2( 10, 0, 110, 30 ), epsilon ), 'Parent child bounds check: ' + node.childBounds.toString() );
    } );

    node.on( 'bounds', function() {
      ok( node.bounds.equalsEpsilon( new dot.Bounds2( 10, 10, 110, 40 ), epsilon ), 'Parent bounds check: ' + node.bounds.toString() );
    } );

    node.on( 'selfBounds', function() {
      ok( false, 'Self bounds should not change for parent node' );
    } );

    rect.on( 'selfBounds', function() {
      ok( rect.selfBounds.equalsEpsilon( new dot.Bounds2( 0, 0, 100, 30 ), epsilon ), 'Self bounds check: ' + rect.selfBounds.toString() );
    } );

    rect.on( 'bounds', function() {
      ok( rect.bounds.equalsEpsilon( new dot.Bounds2( 10, 0, 110, 30 ), epsilon ), 'Bounds check: ' + rect.bounds.toString() );
    } );

    rect.on( 'childBounds', function() {
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

  function compareTrailArrays( a, b ) {
    // defensive copies
    a = a.slice();
    b = b.slice();

    for ( var i = 0; i < a.length; i++ ) {
      // for each A, remove the first matching one in B
      for ( var j = 0; j < b.length; j++ ) {
        if ( a[i].equals( b[j] ) ) {
          b.splice( j, 1 );
          break;
        }
      }
    }

    // now B should be empty
    return b.length === 0;
  }

  test( 'getTrails/getUniqueTrail', function() {
    var a = new scenery.Node();
    var b = new scenery.Node();
    var c = new scenery.Node();
    var d = new scenery.Node();
    var e = new scenery.Node();

    // DAG-like structure
    a.addChild( b );
    a.addChild( c );
    b.addChild( d );
    c.addChild( d );
    c.addChild( e );

    // getUniqueTrail()
    window.assert && throws( function() { d.getUniqueTrail(); }, 'D has no unique trail, since there are two' );
    ok( a.getUniqueTrail().equals( new scenery.Trail( [ a ] ) ), 'a.getUniqueTrail()' );
    ok( b.getUniqueTrail().equals( new scenery.Trail( [ a, b ] ) ), 'b.getUniqueTrail()' );
    ok( c.getUniqueTrail().equals( new scenery.Trail( [ a, c ] ) ), 'c.getUniqueTrail()' );
    ok( e.getUniqueTrail().equals( new scenery.Trail( [ a, c, e ] ) ), 'e.getUniqueTrail()' );

    // getTrails()
    var trails;
    trails = a.getTrails();
    ok( trails.length === 1 && trails[0].equals( new scenery.Trail( [ a ] ) ), 'a.getTrails()' );
    trails = b.getTrails();
    ok( trails.length === 1 && trails[0].equals( new scenery.Trail( [ a, b ] ) ), 'b.getTrails()' );
    trails = c.getTrails();
    ok( trails.length === 1 && trails[0].equals( new scenery.Trail( [ a, c ] ) ), 'c.getTrails()' );
    trails = d.getTrails();
    ok( trails.length === 2 && compareTrailArrays( trails, [ new scenery.Trail( [ a, b, d ] ), new scenery.Trail( [ a, c, d ] ) ]), 'd.getTrails()' );
    trails = e.getTrails();
    ok( trails.length === 1 && trails[0].equals( new scenery.Trail( [ a, c, e ] ) ), 'e.getTrails()' );

    // getUniqueTrail( predicate )
    window.assert && throws( function() { e.getUniqueTrail( function( node ) { return false; } ); }, 'Fails on false predicate' );
    window.assert && throws( function() { e.getUniqueTrail( function( node ) { return false; } ); }, 'Fails on false predicate' );
    ok( e.getUniqueTrail( function( node ) { return node === a; } ).equals( new scenery.Trail( [ a, c, e ] ) ) );
    ok( e.getUniqueTrail( function( node ) { return node === c; } ).equals( new scenery.Trail( [ c, e ] ) ) );
    ok( e.getUniqueTrail( function( node ) { return node === e; } ).equals( new scenery.Trail( [ e ] ) ) );
    ok( d.getUniqueTrail( function( node ) { return node === b; } ).equals( new scenery.Trail( [ b, d ] ) ) );
    ok( d.getUniqueTrail( function( node ) { return node === c; } ).equals( new scenery.Trail( [ c, d ] ) ) );
    ok( d.getUniqueTrail( function( node ) { return node === d; } ).equals( new scenery.Trail( [ d ] ) ) );

    // getTrails( predicate )
    trails = d.getTrails( function( node ) { return false; } );
    ok( trails.length === 0 );
    trails = d.getTrails( function( node ) { return true; } );
    ok( compareTrailArrays( trails, [
      new scenery.Trail( [ a, b, d ] ),
      new scenery.Trail( [ b, d ] ),
      new scenery.Trail( [ a, c, d ] ),
      new scenery.Trail( [ c, d ] ),
      new scenery.Trail( [ d ] )
    ] ) );
    trails = d.getTrails( function( node ) { return node === a; } );
    ok( compareTrailArrays( trails, [
      new scenery.Trail( [ a, b, d ] ),
      new scenery.Trail( [ a, c, d ] )
    ] ) );
    trails = d.getTrails( function( node ) { return node === b; } );
    ok( compareTrailArrays( trails, [
      new scenery.Trail( [ b, d ] )
    ] ) );
    trails = d.getTrails( function( node ) { return node.parents.length === 1; } );
    ok( compareTrailArrays( trails, [
      new scenery.Trail( [ b, d ] ),
      new scenery.Trail( [ c, d ] )
    ] ) );
  } );

  test( 'getLeafTrails', function() {
    var a = new scenery.Node();
    var b = new scenery.Node();
    var c = new scenery.Node();
    var d = new scenery.Node();
    var e = new scenery.Node();

    // DAG-like structure
    a.addChild( b );
    a.addChild( c );
    b.addChild( d );
    c.addChild( d );
    c.addChild( e );

    // getUniqueLeafTrail()
    window.assert && throws( function() { a.getUniqueLeafTrail(); }, 'A has no unique leaf trail, since there are three' );
    ok( b.getUniqueLeafTrail().equals( new scenery.Trail( [ b, d ] ) ), 'a.getUniqueLeafTrail()' );
    ok( d.getUniqueLeafTrail().equals( new scenery.Trail( [ d ] ) ), 'b.getUniqueLeafTrail()' );
    ok( e.getUniqueLeafTrail().equals( new scenery.Trail( [ e ] ) ), 'c.getUniqueLeafTrail()' );

    // getLeafTrails()
    var trails;
    trails = a.getLeafTrails();
    ok( trails.length === 3 && compareTrailArrays( trails, [
      new scenery.Trail( [ a, b, d ] ),
      new scenery.Trail( [ a, c, d ] ),
      new scenery.Trail( [ a, c, e ] )
    ] ), 'a.getLeafTrails()' );
    trails = b.getLeafTrails();
    ok( trails.length === 1 && trails[0].equals( new scenery.Trail( [ b, d ] ) ), 'b.getLeafTrails()' );
    trails = c.getLeafTrails();
    ok( trails.length === 2 && compareTrailArrays( trails, [
      new scenery.Trail( [ c, d ] ),
      new scenery.Trail( [ c, e ] )
    ] ), 'c.getLeafTrails()' );
    trails = d.getLeafTrails();
    ok( trails.length === 1 && trails[0].equals( new scenery.Trail( [ d ] ) ), 'd.getLeafTrails()' );
    trails = e.getLeafTrails();
    ok( trails.length === 1 && trails[0].equals( new scenery.Trail( [ e ] ) ), 'e.getLeafTrails()' );

    // getUniqueLeafTrail( predicate )
    window.assert && throws( function() { e.getUniqueLeafTrail( function( node ) { return false; } ); }, 'Fails on false predicate' );
    window.assert && throws( function() { a.getUniqueLeafTrail( function( node ) { return true; } ); }, 'Fails on multiples' );
    ok( a.getUniqueLeafTrail( function( node ) { return node === e; } ).equals( new scenery.Trail( [ a, c, e ] ) ) );

    // getLeafTrails( predicate )
    trails = a.getLeafTrails( function( node ) { return false; } );
    ok( trails.length === 0 );
    trails = a.getLeafTrails( function( node ) { return true; } );
    ok( compareTrailArrays( trails, [
      new scenery.Trail( [ a ] ),
      new scenery.Trail( [ a, b ] ),
      new scenery.Trail( [ a, b, d ] ),
      new scenery.Trail( [ a, c ] ),
      new scenery.Trail( [ a, c, d ] ),
      new scenery.Trail( [ a, c, e ] )
    ] ) );

    // getLeafTrailsTo( node )
    trails = a.getLeafTrailsTo( d );
    ok( compareTrailArrays( trails, [
      new scenery.Trail( [ a, b, d ] ),
      new scenery.Trail( [ a, c, d ] )
    ] ) );
  } );

  test( 'Line stroked bounds', function() {
    var line = new scenery.Line( 0, 0, 50, 0, { stroke: 'red', lineWidth: 5 } );

    var positions = [
      { x1: 50, y1: 0, x2: 0, y2: 0 },
      { x1: 0, y1: 50, x2: 0, y2: 0 },
      { x1: 0, y1: 0, x2: 50, y2: 0 },
      { x1: 0, y1: 0, x2: 0, y2: 50 },
      { x1: 50, y1: 10, x2: 0, y2: 0 },
      { x1: 10, y1: 50, x2: 0, y2: 0 },
      { x1: 0, y1: 0, x2: 50, y2: 10 },
      { x1: 0, y1: 0, x2: 10, y2: 50 },
      { x1: 50, y1: -10, x2: 0, y2: 0 },
      { x1: -10, y1: 50, x2: 0, y2: 0 },
      { x1: 0, y1: 0, x2: 50, y2: -10 },
      { x1: 0, y1: 0, x2: -10, y2: 50 },
      { x1: 50, y1: 0, x2: 0, y2: 10 },
      { x1: 0, y1: 50, x2: 10, y2: 0 },
      { x1: 0, y1: 10, x2: 50, y2: 0 },
      { x1: 10, y1: 0, x2: 0, y2: 50 },
      { x1: 50, y1: 0, x2: 0, y2: -10 },
      { x1: 0, y1: 50, x2: -10, y2: 0 },
      { x1: 0, y1: -10, x2: 50, y2: 0 },
      { x1: -10, y1: 0, x2: 0, y2: 50 }
    ];

    var caps = [
      'round',
      'butt',
      'square'
    ];

    _.each( positions, function( position ) {
      line.mutate( position );
      // line.setLine( position.x1, position.y1, position.x2, position.y2 );
      _.each( caps, function( cap ) {
        line.lineCap = cap;

        ok( line.bounds.equalsEpsilon( line.getShape().getStrokedShape( line.getLineStyles() ).bounds, 0.0001 ),
          'Line stroked bounds with ' + JSON.stringify( position ) + ' and ' + cap + ' ' + line.bounds.toString() );
      } );
    } );
  } );

  test( 'Color listener non-memory-leak-ness', function() {
    var scene = new scenery.Node();
    var display = new scenery.Display( scene, { width: 20, height: 20 } );
    display.updateDisplay();
    var node = new scenery.Node();
    scene.addChild( node );

    var color = new scenery.Color( 255, 255, 0 );
    equal( color.getListenerCount(), 0, 'Initial colors should have no listeners' );

    var rect = new scenery.Rectangle( 0, 0, 50, 50, { fill: color } );
    node.addChild( rect );
    equal( color.getListenerCount(), 0, 'Still no listeners until updateDisplay' );

    display.updateDisplay();
    equal( color.getListenerCount(), 1, 'One listener for the rectangle' );

    var circle = new scenery.Circle( 40, { fill: color, stroke: color } );
    node.addChild( circle );
    display.updateDisplay();
    equal( color.getListenerCount(), 3, 'One listener for the rectangle, two for the circle (stroke/fill)' );

    circle.stroke = null;
    equal( color.getListenerCount(), 2, 'One listener for the rectangle, one for the circle (fill)' );

    rect.addChild( circle );
    display.updateDisplay();
    equal( color.getListenerCount(), 3, 'One listener for the rectangle, two for the circle (two instances)' );

    node.removeAllChildren();
    display.updateDisplay();
    equal( color.getListenerCount(), 0, 'Nothing else attached' );
  } );

  test( 'maxWidth/maxHeight for Node', function() {
    var rect = new scenery.Rectangle( 0, 0, 100, 50, { fill: 'red' } );
    var node = new scenery.Node( { children: [ rect ] } );

    ok( node.bounds.equals( new dot.Bounds2( 0, 0, 100, 50 ) ), 'Initial bounds' );

    node.maxWidth = 50;

    ok( node.bounds.equals( new dot.Bounds2( 0, 0, 50, 25 ) ), 'Halved transform after max width of half' );

    node.maxWidth = 120;

    ok( node.bounds.equals( new dot.Bounds2( 0, 0, 100, 50 ) ), 'Back to normal after a big max width' );

    node.scale( 2 );

    ok( node.bounds.equals( new dot.Bounds2( 0, 0, 200, 100 ) ), 'Scale up should be unaffected' );

    node.maxWidth = 25;

    ok( node.bounds.equals( new dot.Bounds2( 0, 0, 50, 25 ) ), 'Scaled back down with both applied' );

    node.maxWidth = null;

    ok( node.bounds.equals( new dot.Bounds2( 0, 0, 200, 100 ) ), 'Without maxWidth' );

    node.scale( 0.5 );

    ok( node.bounds.equals( new dot.Bounds2( 0, 0, 100, 50 ) ), 'Back to normal' );

    node.left = 50;

    ok( node.bounds.equals( new dot.Bounds2( 50, 0, 150, 50 ) ), 'After a translation' );

    node.maxWidth = 50;

    ok( node.bounds.equals( new dot.Bounds2( 50, 0, 100, 25 ) ), 'maxWidth being applied after a translation, in local frame' );

    rect.rectWidth = 200;

    ok( node.bounds.equals( new dot.Bounds2( 50, 0, 100, 12.5 ) ), 'Now with a bigger rectangle' );

    rect.rectWidth = 100;
    node.maxWidth = null;

    ok( node.bounds.equals( new dot.Bounds2( 50, 0, 150, 50 ) ), 'Back to a translation' );

    rect.maxWidth = 50;

    ok( node.bounds.equals( new dot.Bounds2( 50, 0, 100, 25 ) ), 'After maxWidth A' );

    rect.maxHeight = 12.5;

    ok( node.bounds.equals( new dot.Bounds2( 50, 0, 75, 12.5 ) ), 'After maxHeight A' );
  } );

  test( 'Spacers', function() {
    var spacer = new scenery.Spacer( 100, 50, { x: 50 } );
    ok( spacer.bounds.equals( new dot.Bounds2( 50, 0, 150, 50 ) ), 'Spacer bounds with translation' );

    var hstrut = new scenery.HStrut( 100, { y: 50 } );
    ok( hstrut.bounds.equals( new dot.Bounds2( 0, 50, 100, 50 ) ), 'HStrut bounds with translation' );

    var vstrut = new scenery.VStrut( 100, { x: 50 } );
    ok( vstrut.bounds.equals( new dot.Bounds2( 50, 0, 50, 100 ) ), 'VStrut bounds with translation' );

    throws( function() {
      spacer.addChild( new scenery.Node() );
    }, 'No way to add children to Spacer' );

    throws( function() {
      hstrut.addChild( new scenery.Node() );
    }, 'No way to add children to HStrut' );

    throws( function() {
      vstrut.addChild( new scenery.Node() );
    }, 'No way to add children to VStrut' );
  } );

  test( 'Renderer Summary', function() {
    var canvasNode = new scenery.CanvasNode( { canvasBounds: new dot.Bounds2( 0, 0, 10, 10 ) } );
    var webglNode = new scenery.WebGLNode( function() {}, { canvasBounds: new dot.Bounds2( 0, 0, 10, 10 ) } );
    var rect = new scenery.Rectangle( 0, 0, 100, 50 );
    var node = new scenery.Node( { children: [ canvasNode, webglNode, rect ] } );
    var emptyNode = new scenery.Node();

    ok( canvasNode._rendererSummary.isSubtreeFullyCompatible( scenery.Renderer.bitmaskCanvas ), 'CanvasNode fully compatible: Canvas' );
    ok( !canvasNode._rendererSummary.isSubtreeFullyCompatible( scenery.Renderer.bitmaskSVG ), 'CanvasNode not fully compatible: SVG' );
    ok( canvasNode._rendererSummary.isSubtreeContainingCompatible( scenery.Renderer.bitmaskCanvas ), 'CanvasNode partially compatible: Canvas' );
    ok( !canvasNode._rendererSummary.isSubtreeContainingCompatible( scenery.Renderer.bitmaskSVG ), 'CanvasNode not partially compatible: SVG' );
    ok( canvasNode._rendererSummary.isSingleCanvasSupported(), 'CanvasNode supports single Canvas' );
    ok( !canvasNode._rendererSummary.isSingleSVGSupported(), 'CanvasNode does not support single SVG' );
    ok( !canvasNode._rendererSummary.isNotPainted(), 'CanvasNode is painted' );
    ok( canvasNode._rendererSummary.areBoundsValid(), 'CanvasNode has valid bounds' );

    ok( webglNode._rendererSummary.isSubtreeFullyCompatible( scenery.Renderer.bitmaskWebGL ), 'WebGLNode fully compatible: WebGL' );
    ok( !webglNode._rendererSummary.isSubtreeFullyCompatible( scenery.Renderer.bitmaskSVG ), 'WebGLNode not fully compatible: SVG' );
    ok( webglNode._rendererSummary.isSubtreeContainingCompatible( scenery.Renderer.bitmaskWebGL ), 'WebGLNode partially compatible: WebGL' );
    ok( !webglNode._rendererSummary.isSubtreeContainingCompatible( scenery.Renderer.bitmaskSVG ), 'WebGLNode not partially compatible: SVG' );
    ok( !webglNode._rendererSummary.isSingleCanvasSupported(), 'WebGLNode does not support single Canvas' );
    ok( !webglNode._rendererSummary.isSingleSVGSupported(), 'WebGLNode does not support single SVG' );
    ok( !webglNode._rendererSummary.isNotPainted(), 'WebGLNode is painted' );
    ok( webglNode._rendererSummary.areBoundsValid(), 'WebGLNode has valid bounds' );

    ok( rect._rendererSummary.isSubtreeFullyCompatible( scenery.Renderer.bitmaskCanvas ), 'Rectangle fully compatible: Canvas' );
    ok( rect._rendererSummary.isSubtreeFullyCompatible( scenery.Renderer.bitmaskSVG ), 'Rectangle fully compatible: SVG' );
    ok( rect._rendererSummary.isSubtreeContainingCompatible( scenery.Renderer.bitmaskCanvas ), 'Rectangle partially compatible: Canvas' );
    ok( rect._rendererSummary.isSubtreeContainingCompatible( scenery.Renderer.bitmaskSVG ), 'Rectangle partially compatible: SVG' );
    ok( rect._rendererSummary.isSingleCanvasSupported(), 'Rectangle does support single Canvas' );
    ok( rect._rendererSummary.isSingleSVGSupported(), 'Rectangle does support single SVG' );
    ok( !rect._rendererSummary.isNotPainted(), 'Rectangle is painted' );
    ok( rect._rendererSummary.areBoundsValid(), 'Rectangle has valid bounds' );

    ok( !node._rendererSummary.isSubtreeFullyCompatible( scenery.Renderer.bitmaskCanvas ), 'Container node fully compatible: Canvas' );
    ok( !node._rendererSummary.isSubtreeFullyCompatible( scenery.Renderer.bitmaskSVG ), 'Container node not fully compatible: SVG' );
    ok( node._rendererSummary.isSubtreeContainingCompatible( scenery.Renderer.bitmaskCanvas ), 'Container node partially compatible: Canvas' );
    ok( node._rendererSummary.isSubtreeContainingCompatible( scenery.Renderer.bitmaskSVG ), 'Container node partially compatible: SVG' );
    ok( !node._rendererSummary.isSingleCanvasSupported(), 'Container node does not support single Canvas' );
    ok( !node._rendererSummary.isSingleSVGSupported(), 'Container node does not support single SVG' );
    ok( !node._rendererSummary.isNotPainted(), 'Container node is painted' );
    ok( node._rendererSummary.areBoundsValid(), 'Container node has valid bounds' );

    ok( emptyNode._rendererSummary.isSubtreeFullyCompatible( scenery.Renderer.bitmaskCanvas ), 'Empty node fully compatible: Canvas' );
    ok( emptyNode._rendererSummary.isSubtreeFullyCompatible( scenery.Renderer.bitmaskSVG ), 'Empty node fully compatible: SVG' );
    ok( !emptyNode._rendererSummary.isSubtreeContainingCompatible( scenery.Renderer.bitmaskCanvas ), 'Empty node partially compatible: Canvas' );
    ok( !emptyNode._rendererSummary.isSubtreeContainingCompatible( scenery.Renderer.bitmaskSVG ), 'Empty node partially compatible: SVG' );
    ok( emptyNode._rendererSummary.isSingleCanvasSupported(), 'Empty node supports single Canvas' );
    ok( emptyNode._rendererSummary.isSingleSVGSupported(), 'Empty node supports single SVG' );
    ok( emptyNode._rendererSummary.isNotPainted(), 'Empty node is not painted' );
    ok( emptyNode._rendererSummary.areBoundsValid(), 'Empty node has valid bounds' );
  } );

  /* eslint-enable */
})();
