// Copyright 2017, University of Colorado Boulder

/**
 * Trail tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var Bounds2 = require( 'DOT/Bounds2' );
  var CanvasNode = require( 'SCENERY/nodes/CanvasNode' );
  var Color = require( 'SCENERY/util/Color' );
  var Display = require( 'SCENERY/display/Display' );
  var HStrut = require( 'SCENERY/nodes/HStrut' );
  var Line = require( 'SCENERY/nodes/Line' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var Shape = require( 'KITE/Shape' );
  var Spacer = require( 'SCENERY/nodes/Spacer' );
  var Text = require( 'SCENERY/nodes/Text' );
  var TextBounds = require( 'SCENERY/util/TextBounds' );
  var Trail = require( 'SCENERY/util/Trail' );
  var TrailPointer = require( 'SCENERY/util/TrailPointer' );
  var Util = require( 'SCENERY/util/Util' );
  var Vector2 = require( 'DOT/Vector2' );
  var VStrut = require( 'SCENERY/nodes/VStrut' );
  var WebGLNode = require( 'SCENERY/nodes/WebGLNode' );

  QUnit.module( 'Trail' );

  /* eslint-disable no-undef */

  function equalsApprox( assert, a, b, message ) { // eslint-disable-line no-unused-vars
    assert.ok( Math.abs( a - b ) < 0.0000001, ( message ? message + ': ' : '' ) + a + ' =? ' + b );
  }

  function createTestNodeTree() { // eslint-disable-line no-unused-vars
    var node = new Node();
    node.addChild( new Node() );
    node.addChild( new Node() );
    node.addChild( new Node() );

    node.children[ 0 ].addChild( new Node() );
    node.children[ 0 ].addChild( new Node() );
    node.children[ 0 ].addChild( new Node() );
    node.children[ 0 ].addChild( new Node() );
    node.children[ 0 ].addChild( new Node() );

    node.children[ 0 ].children[ 1 ].addChild( new Node() );
    node.children[ 0 ].children[ 3 ].addChild( new Node() );
    node.children[ 0 ].children[ 3 ].addChild( new Node() );

    node.children[ 0 ].children[ 3 ].children[ 0 ].addChild( new Node() );

    return node;
  }

  QUnit.test( 'Dirty bounds propagation test', function( assert ) {
    var node = createTestNodeTree();

    node.validateBounds();

    assert.ok( !node._childBoundsDirty );

    node.children[ 0 ].children[ 3 ].children[ 0 ].invalidateBounds();

    assert.ok( node._childBoundsDirty );
  } );

  QUnit.test( 'Canvas 2D Context and Features', function( assert ) {
    var canvas = document.createElement( 'canvas' );
    var context = canvas.getContext( '2d' );

    assert.ok( context, 'context' );

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
      assert.ok( context[ method ] !== undefined, 'context.' + method );
    } );
  } );

  QUnit.test( 'Trail next/previous', function( assert ) {
    var node = createTestNodeTree();

    // walk it forward
    var trail = new Trail( [ node ] );
    assert.equal( 1, trail.length );
    trail = trail.next();
    assert.equal( 2, trail.length );
    trail = trail.next();
    assert.equal( 3, trail.length );
    trail = trail.next();
    assert.equal( 3, trail.length );
    trail = trail.next();
    assert.equal( 4, trail.length );
    trail = trail.next();
    assert.equal( 3, trail.length );
    trail = trail.next();
    assert.equal( 3, trail.length );
    trail = trail.next();
    assert.equal( 4, trail.length );
    trail = trail.next();
    assert.equal( 5, trail.length );
    trail = trail.next();
    assert.equal( 4, trail.length );
    trail = trail.next();
    assert.equal( 3, trail.length );
    trail = trail.next();
    assert.equal( 2, trail.length );
    trail = trail.next();
    assert.equal( 2, trail.length );

    // make sure walking off the end gives us null
    assert.equal( null, trail.next() );

    trail = trail.previous();
    assert.equal( 2, trail.length );
    trail = trail.previous();
    assert.equal( 3, trail.length );
    trail = trail.previous();
    assert.equal( 4, trail.length );
    trail = trail.previous();
    assert.equal( 5, trail.length );
    trail = trail.previous();
    assert.equal( 4, trail.length );
    trail = trail.previous();
    assert.equal( 3, trail.length );
    trail = trail.previous();
    assert.equal( 3, trail.length );
    trail = trail.previous();
    assert.equal( 4, trail.length );
    trail = trail.previous();
    assert.equal( 3, trail.length );
    trail = trail.previous();
    assert.equal( 3, trail.length );
    trail = trail.previous();
    assert.equal( 2, trail.length );
    trail = trail.previous();
    assert.equal( 1, trail.length );

    // make sure walking off the start gives us null
    assert.equal( null, trail.previous() );
  } );

  QUnit.test( 'Trail comparison', function( assert ) {
    var node = createTestNodeTree();

    // get a list of all trails in render order
    var trails = [];
    var currentTrail = new Trail( node ); // start at the first node

    while ( currentTrail ) {
      trails.push( currentTrail );
      currentTrail = currentTrail.next();
    }

    assert.equal( 13, trails.length, 'Trail for each node' );

    for ( var i = 0; i < trails.length; i++ ) {
      for ( var j = i; j < trails.length; j++ ) {
        var comparison = trails[ i ].compare( trails[ j ] );

        // make sure that every trail compares as expected (0 and they are equal, -1 and i < j)
        assert.equal( i === j ? 0 : ( i < j ? -1 : 1 ), comparison, i + ',' + j );
      }
    }
  } );

  QUnit.test( 'Trail eachTrailBetween', function( assert ) {
    var node = createTestNodeTree();

    // get a list of all trails in render order
    var trails = [];
    var currentTrail = new Trail( node ); // start at the first node

    while ( currentTrail ) {
      trails.push( currentTrail );
      currentTrail = currentTrail.next();
    }

    assert.equal( 13, trails.length, 'Trails: ' + _.map( trails, function( trail ) { return trail.toString(); } ).join( '\n' ) );

    for ( var i = 0; i < trails.length; i++ ) {
      for ( var j = i; j < trails.length; j++ ) {
        var inclusiveList = [];
        Trail.eachTrailBetween( trails[ i ], trails[ j ], function( trail ) {
          inclusiveList.push( trail.copy() );
        }, false, node );
        var trailString = i + ',' + j + ' ' + trails[ i ].toString() + ' to ' + trails[ j ].toString();
        assert.ok( inclusiveList[ 0 ].equals( trails[ i ] ), 'inclusive start on ' + trailString + ' is ' + inclusiveList[ 0 ].toString() );
        assert.ok( inclusiveList[ inclusiveList.length - 1 ].equals( trails[ j ] ), 'inclusive end on ' + trailString + 'is ' + inclusiveList[ inclusiveList.length - 1 ].toString() );
        assert.equal( inclusiveList.length, j - i + 1, 'inclusive length on ' + trailString + ' is ' + inclusiveList.length + ', ' + _.map( inclusiveList, function( trail ) { return trail.toString(); } ).join( '\n' ) );

        if ( i < j ) {
          var exclusiveList = [];
          Trail.eachTrailBetween( trails[ i ], trails[ j ], function( trail ) {
            exclusiveList.push( trail.copy() );
          }, true, node );
          assert.equal( exclusiveList.length, j - i - 1, 'exclusive length on ' + i + ',' + j );
        }
      }
    }
  } );

  QUnit.test( 'depthFirstUntil depthFirstUntil with subtree skipping', function( assert ) {
    var node = createTestNodeTree();
    node.children[ 0 ].children[ 2 ].visible = false;
    node.children[ 0 ].children[ 3 ].visible = false;
    new TrailPointer( new Trail( node ), true ).depthFirstUntil( new TrailPointer( new Trail( node ), false ), function( pointer ) {
      if ( !pointer.trail.lastNode().isVisible() ) {
        // should skip
        return true;
      }
      assert.ok( pointer.trail.isVisible(), 'Trail visibility for ' + pointer.trail.toString() );
    }, false );
  } );

  QUnit.test( 'Trail eachTrailUnder with subtree skipping', function( assert ) {
    var node = createTestNodeTree();
    node.children[ 0 ].children[ 2 ].visible = false;
    node.children[ 0 ].children[ 3 ].visible = false;
    new Trail( node ).eachTrailUnder( function( trail ) {
      if ( !trail.lastNode().isVisible() ) {
        // should skip
        return true;
      }
      assert.ok( trail.isVisible(), 'Trail visibility for ' + trail.toString() );
    } );
  } );

  QUnit.test( 'TrailPointer render comparison', function( assert ) {
    var node = createTestNodeTree();

    assert.equal( 0, new TrailPointer( node.getUniqueTrail(), true ).compareRender( new TrailPointer( node.getUniqueTrail(), true ) ), 'Same before pointer' );
    assert.equal( 0, new TrailPointer( node.getUniqueTrail(), false ).compareRender( new TrailPointer( node.getUniqueTrail(), false ) ), 'Same after pointer' );
    assert.equal( -1, new TrailPointer( node.getUniqueTrail(), true ).compareRender( new TrailPointer( node.getUniqueTrail(), false ) ), 'Same node before/after root' );
    assert.equal( -1, new TrailPointer( node.children[ 0 ].getUniqueTrail(), true ).compareRender( new TrailPointer( node.children[ 0 ].getUniqueTrail(), false ) ), 'Same node before/after nonroot' );
    assert.equal( 0, new TrailPointer( node.children[ 0 ].children[ 1 ].children[ 0 ].getUniqueTrail(), false ).compareRender( new TrailPointer( node.children[ 0 ].children[ 2 ].getUniqueTrail(), true ) ), 'Equivalence of before/after' );

    // all pointers in the render order
    var pointers = [
      new TrailPointer( node.getUniqueTrail(), true ),
      new TrailPointer( node.getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 0 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 0 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 1 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 1 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 1 ].children[ 0 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 1 ].children[ 0 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 2 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 2 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].children[ 0 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].children[ 0 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 1 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 1 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 4 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 4 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 1 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 1 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 2 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 2 ].getUniqueTrail(), false )
    ];

    // compare the pointers. different ones can be equal if they represent the same place, so we only check if they compare differently
    for ( var i = 0; i < pointers.length; i++ ) {
      for ( var j = i; j < pointers.length; j++ ) {
        var comparison = pointers[ i ].compareRender( pointers[ j ] );

        if ( comparison === -1 ) {
          assert.ok( i < j, i + ',' + j );
        }
        if ( comparison === 1 ) {
          assert.ok( i > j, i + ',' + j );
        }
      }
    }
  } );

  QUnit.test( 'TrailPointer nested comparison and fowards/backwards', function( assert ) {
    var node = createTestNodeTree();

    // all pointers in the nested order
    var pointers = [
      new TrailPointer( node.getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 0 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 0 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 1 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 1 ].children[ 0 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 1 ].children[ 0 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 1 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 2 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 2 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].children[ 0 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].children[ 0 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 0 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 1 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].children[ 1 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 3 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].children[ 4 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 0 ].children[ 4 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 0 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 1 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 1 ].getUniqueTrail(), false ),
      new TrailPointer( node.children[ 2 ].getUniqueTrail(), true ),
      new TrailPointer( node.children[ 2 ].getUniqueTrail(), false ),
      new TrailPointer( node.getUniqueTrail(), false )
    ];

    // exhaustively verify the ordering between each ordered pair
    for ( var i = 0; i < pointers.length; i++ ) {
      for ( var j = i; j < pointers.length; j++ ) {
        var comparison = pointers[ i ].compareNested( pointers[ j ] );

        // make sure that every pointer compares as expected (0 and they are equal, -1 and i < j)
        assert.equal( comparison, i === j ? 0 : ( i < j ? -1 : 1 ), 'compareNested: ' + i + ',' + j );
      }
    }

    // verify forwards and backwards, as well as copy constructors
    for ( i = 1; i < pointers.length; i++ ) {
      var a = pointers[ i - 1 ];
      var b = pointers[ i ];

      var forwardsCopy = a.copy();
      forwardsCopy.nestedForwards();
      assert.equal( forwardsCopy.compareNested( b ), 0, 'forwardsPointerCheck ' + ( i - 1 ) + ' to ' + i );

      var backwardsCopy = b.copy();
      backwardsCopy.nestedBackwards();
      assert.equal( backwardsCopy.compareNested( a ), 0, 'backwardsPointerCheck ' + i + ' to ' + ( i - 1 ) );
    }

    // exhaustively check depthFirstUntil inclusive
    for ( i = 0; i < pointers.length; i++ ) {
      for ( j = i + 1; j < pointers.length; j++ ) {
        // i < j guaranteed
        var contents = [];
        pointers[ i ].depthFirstUntil( pointers[ j ], function( pointer ) { contents.push( pointer.copy() ); }, false );
        assert.equal( contents.length, j - i + 1, 'depthFirstUntil inclusive ' + i + ',' + j + ' count check' );

        // do an actual pointer to pointer comparison
        var isOk = true;
        for ( var k = 0; k < contents.length; k++ ) {
          comparison = contents[ k ].compareNested( pointers[ i + k ] );
          if ( comparison !== 0 ) {
            assert.equal( comparison, 0, 'depthFirstUntil inclusive ' + i + ',' + j + ',' + k + ' comparison check ' + contents[ k ].trail.indices.join() + ' - ' + pointers[ i + k ].trail.indices.join() );
            isOk = false;
          }
        }
        assert.ok( isOk, 'depthFirstUntil inclusive ' + i + ',' + j + ' comparison check' );
      }
    }

    // exhaustively check depthFirstUntil exclusive
    for ( i = 0; i < pointers.length; i++ ) {
      for ( j = i + 1; j < pointers.length; j++ ) {
        // i < j guaranteed
        contents = [];
        pointers[ i ].depthFirstUntil( pointers[ j ], function( pointer ) { contents.push( pointer.copy() ); }, true );
        assert.equal( contents.length, j - i - 1, 'depthFirstUntil exclusive ' + i + ',' + j + ' count check' );

        // do an actual pointer to pointer comparison
        isOk = true;
        for ( k = 0; k < contents.length; k++ ) {
          comparison = contents[ k ].compareNested( pointers[ i + k + 1 ] );
          if ( comparison !== 0 ) {
            assert.equal( comparison, 0, 'depthFirstUntil exclusive ' + i + ',' + j + ',' + k + ' comparison check ' + contents[ k ].trail.indices.join() + ' - ' + pointers[ i + k ].trail.indices.join() );
            isOk = false;
          }
        }
        assert.ok( isOk, 'depthFirstUntil exclusive ' + i + ',' + j + ' comparison check' );
      }
    }
  } );

  // QUnit.test( 'TrailInterval', function(assert) {
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
  //       var interval = new TrailInterval( trails[i], trails[j] );
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
  //            assert.equal( a.exclusiveContains( trail ) || b.exclusiveContains( trail ), union.exclusiveContains( trail ), msg );
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
  //         assert.ok( containsAnything && wouldBeBadUnion, 'Not a bad union?: ' + a.toString() + ' and ' + b.toString() + ' with union ' + union.toString() );
  //       }
  //     }
  //   }
  // } );

  QUnit.test( 'Text width measurement in canvas', function( assert ) {
    var canvas = document.createElement( 'canvas' );
    var context = canvas.getContext( '2d' );
    var metrics = context.measureText( 'Hello World' );
    assert.ok( metrics.width, 'metrics.width' );
  } );

  QUnit.test( 'Sceneless node handling', function( assert ) {
    var a = new Path( null );
    var b = new Path( null );
    var c = new Path( null );

    a.setShape( Shape.rectangle( 0, 0, 20, 20 ) );
    c.setShape( Shape.rectangle( 10, 10, 30, 30 ) );

    a.addChild( b );
    b.addChild( c );

    a.validateBounds();

    a.removeChild( b );
    c.addChild( a );

    b.validateBounds();

    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'Correct bounds on rectangle', function( assert ) {
    var rectBounds = Util.canvasAccurateBounds( function( context ) { context.fillRect( 100, 100, 200, 200 ); } );
    assert.ok( Math.abs( rectBounds.minX - 100 ) < 0.01, rectBounds.minX );
    assert.ok( Math.abs( rectBounds.minY - 100 ) < 0.01, rectBounds.minY );
    assert.ok( Math.abs( rectBounds.maxX - 300 ) < 0.01, rectBounds.maxX );
    assert.ok( Math.abs( rectBounds.maxY - 300 ) < 0.01, rectBounds.maxY );
  } );

  QUnit.test( 'Consistent and precise bounds range on Text', function( assert ) {
    var textBounds = Util.canvasAccurateBounds( function( context ) { context.fillText( 'test string', 0, 0 ); } );
    assert.ok( textBounds.isConsistent, textBounds.toString() );

    // precision of 0.001 (or lower given different parameters) is possible on non-Chome browsers (Firefox, IE9, Opera)
    assert.ok( textBounds.precision < 0.15, 'precision: ' + textBounds.precision );
  } );

  QUnit.test( 'Consistent and precise bounds range on Text', function( assert ) {
    var text = new Text( '0\u0489' );
    var textBounds = TextBounds.accurateCanvasBounds( text );
    assert.ok( textBounds.isConsistent, textBounds.toString() );

    // precision of 0.001 (or lower given different parameters) is possible on non-Chome browsers (Firefox, IE9, Opera)
    assert.ok( textBounds.precision < 1, 'precision: ' + textBounds.precision );
  } );

  QUnit.test( 'ES5 Setter / Getter tests', function( assert ) {
    var node = new Path( null );
    var fill = '#abcdef';
    node.fill = fill;
    assert.equal( node.fill, fill );
    assert.equal( node.getFill(), fill );

    var otherNode = new Path( Shape.rectangle( 0, 0, 10, 10 ), { fill: fill } );

    assert.equal( otherNode.fill, fill );
  } );

  QUnit.test( 'Piccolo-like behavior', function( assert ) {
    var node = new Node();

    node.scale( 2 );
    node.translate( 1, 3 );
    node.rotate( Math.PI / 2 );
    node.translate( -31, 21 );

    equalsApprox( assert, node.getMatrix().m00(), 0 );
    equalsApprox( assert, node.getMatrix().m01(), -2 );
    equalsApprox( assert, node.getMatrix().m02(), -40 );
    equalsApprox( assert, node.getMatrix().m10(), 2 );
    equalsApprox( assert, node.getMatrix().m11(), 0 );
    equalsApprox( assert, node.getMatrix().m12(), -56 );

    equalsApprox( assert, node.x, -40 );
    equalsApprox( assert, node.y, -56 );
    equalsApprox( assert, node.rotation, Math.PI / 2 );

    node.translation = new Vector2( -5, 7 );

    equalsApprox( assert, node.getMatrix().m02(), -5 );
    equalsApprox( assert, node.getMatrix().m12(), 7 );

    node.rotation = 1.2;

    equalsApprox( assert, node.getMatrix().m01(), -1.864078171934453 );

    node.rotation = -0.7;

    equalsApprox( assert, node.getMatrix().m10(), -1.288435374475382 );
  } );

  QUnit.test( 'Setting left/right of node', function( assert ) {
    var node = new Path( Shape.rectangle( -20, -20, 50, 50 ), {
      scale: 2
    } );

    equalsApprox( assert, node.left, -40 );
    equalsApprox( assert, node.right, 60 );

    node.left = 10;
    equalsApprox( assert, node.left, 10 );
    equalsApprox( assert, node.right, 110 );

    node.right = 10;
    equalsApprox( assert, node.left, -90 );
    equalsApprox( assert, node.right, 10 );

    node.centerX = 5;
    equalsApprox( assert, node.centerX, 5 );
    equalsApprox( assert, node.left, -45 );
    equalsApprox( assert, node.right, 55 );
  } );

  QUnit.test( 'Path with empty shape', function( assert ) {
    var scene = new Node();
    var node = new Path( new Shape() );
    scene.addChild( node );
    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'Path with null shape', function( assert ) {
    var scene = new Node();
    var node = new Path( null );
    scene.addChild( node );
    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'Display resize event', function( assert ) {
    var scene = new Node();
    var display = new Display( scene );

    var width;
    var height;
    var count = 0;

    display.on( 'displaySize', function( size ) {
      width = size.width;
      height = size.height;
      count++;
    } );

    display.setWidthHeight( 712, 217 );

    assert.equal( width, 712, 'Scene resize width' );
    assert.equal( height, 217, 'Scene resize height' );
    assert.equal( count, 1, 'Scene resize count' );
  } );

  QUnit.test( 'Bounds events', function( assert ) {
    var node = new Node();
    node.y = 10;

    var rect = new Rectangle( 0, 0, 100, 50, { fill: '#f00' } );
    rect.x = 10; // a transform, so we can verify everything is handled correctly
    node.addChild( rect );

    node.validateBounds();

    var epsilon = 0.0000001;

    node.on( 'childBounds', function() {
      assert.ok( node.childBounds.equalsEpsilon( new Bounds2( 10, 0, 110, 30 ), epsilon ), 'Parent child bounds check: ' + node.childBounds.toString() );
    } );

    node.on( 'bounds', function() {
      assert.ok( node.bounds.equalsEpsilon( new Bounds2( 10, 10, 110, 40 ), epsilon ), 'Parent bounds check: ' + node.bounds.toString() );
    } );

    node.on( 'selfBounds', function() {
      assert.ok( false, 'Self bounds should not change for parent node' );
    } );

    rect.on( 'selfBounds', function() {
      assert.ok( rect.selfBounds.equalsEpsilon( new Bounds2( 0, 0, 100, 30 ), epsilon ), 'Self bounds check: ' + rect.selfBounds.toString() );
    } );

    rect.on( 'bounds', function() {
      assert.ok( rect.bounds.equalsEpsilon( new Bounds2( 10, 0, 110, 30 ), epsilon ), 'Bounds check: ' + rect.bounds.toString() );
    } );

    rect.on( 'childBounds', function() {
      assert.ok( false, 'Child bounds should not change for leaf node' );
    } );

    rect.rectHeight = 30;
    node.validateBounds();
  } );

  QUnit.test( 'Using a color instance', function( assert ) {
    var scene = new Node();

    var rect = new Rectangle( 0, 0, 100, 50 );
    assert.ok( rect.fill === null, 'Always starts with a null fill' );
    scene.addChild( rect );
    var color = new Color( 255, 0, 0 );
    rect.fill = color;
    color.setRGBA( 0, 255, 0, 1 );
  } );

  QUnit.test( 'Bounds and Visible Bounds', function( assert ) {
    var node = new Node();
    var rect = new Rectangle( 0, 0, 100, 50 );
    node.addChild( rect );

    assert.ok( node.visibleBounds.equals( new Bounds2( 0, 0, 100, 50 ) ), 'Visible Bounds Visible' );
    assert.ok( node.bounds.equals( new Bounds2( 0, 0, 100, 50 ) ), 'Complete Bounds Visible' );

    rect.visible = false;

    assert.ok( node.visibleBounds.equals( Bounds2.NOTHING ), 'Visible Bounds Invisible' );
    assert.ok( node.bounds.equals( new Bounds2( 0, 0, 100, 50 ) ), 'Complete Bounds Invisible' );
  } );

  QUnit.test( 'localBounds override', function( assert ) {
    var node = new Node( { y: 5 } );
    var rect = new Rectangle( 0, 0, 100, 50 );
    node.addChild( rect );

    rect.localBounds = new Bounds2( 0, 0, 50, 50 );
    assert.ok( node.localBounds.equals( new Bounds2( 0, 0, 50, 50 ) ), 'localBounds override on self' );
    assert.ok( node.bounds.equals( new Bounds2( 0, 5, 50, 55 ) ), 'localBounds override on self' );

    rect.localBounds = new Bounds2( 0, 0, 50, 100 );
    assert.ok( node.bounds.equals( new Bounds2( 0, 5, 50, 105 ) ), 'localBounds override 2nd on self' );

    // reset local bounds (have them computed again)
    rect.localBounds = null;
    assert.ok( node.bounds.equals( new Bounds2( 0, 5, 100, 55 ) ), 'localBounds override reset on self' );

    node.localBounds = new Bounds2( 0, 0, 50, 200 );
    assert.ok( node.localBounds.equals( new Bounds2( 0, 0, 50, 200 ) ), 'localBounds override on parent' );
    assert.ok( node.bounds.equals( new Bounds2( 0, 5, 50, 205 ) ), 'localBounds override on parent' );
  } );

  function compareTrailArrays( a, b ) {
    // defensive copies
    a = a.slice();
    b = b.slice();

    for ( var i = 0; i < a.length; i++ ) {
      // for each A, remove the first matching one in B
      for ( var j = 0; j < b.length; j++ ) {
        if ( a[ i ].equals( b[ j ] ) ) {
          b.splice( j, 1 );
          break;
        }
      }
    }

    // now B should be empty
    return b.length === 0;
  }

  QUnit.test( 'getTrails/getUniqueTrail', function( assert ) {
    var a = new Node();
    var b = new Node();
    var c = new Node();
    var d = new Node();
    var e = new Node();

    // DAG-like structure
    a.addChild( b );
    a.addChild( c );
    b.addChild( d );
    c.addChild( d );
    c.addChild( e );

    // getUniqueTrail()
    window.assert && assert.throws( function() { d.getUniqueTrail(); }, 'D has no unique trail, since there are two' );
    assert.ok( a.getUniqueTrail().equals( new Trail( [ a ] ) ), 'a.getUniqueTrail()' );
    assert.ok( b.getUniqueTrail().equals( new Trail( [ a, b ] ) ), 'b.getUniqueTrail()' );
    assert.ok( c.getUniqueTrail().equals( new Trail( [ a, c ] ) ), 'c.getUniqueTrail()' );
    assert.ok( e.getUniqueTrail().equals( new Trail( [ a, c, e ] ) ), 'e.getUniqueTrail()' );

    // getTrails()
    var trails;
    trails = a.getTrails();
    assert.ok( trails.length === 1 && trails[ 0 ].equals( new Trail( [ a ] ) ), 'a.getTrails()' );
    trails = b.getTrails();
    assert.ok( trails.length === 1 && trails[ 0 ].equals( new Trail( [ a, b ] ) ), 'b.getTrails()' );
    trails = c.getTrails();
    assert.ok( trails.length === 1 && trails[ 0 ].equals( new Trail( [ a, c ] ) ), 'c.getTrails()' );
    trails = d.getTrails();
    assert.ok( trails.length === 2 && compareTrailArrays( trails, [ new Trail( [ a, b, d ] ), new Trail( [ a, c, d ] ) ] ), 'd.getTrails()' );
    trails = e.getTrails();
    assert.ok( trails.length === 1 && trails[ 0 ].equals( new Trail( [ a, c, e ] ) ), 'e.getTrails()' );

    // getUniqueTrail( predicate )
    window.assert && assert.throws( function() { e.getUniqueTrail( function( node ) { return false; } ); }, 'Fails on false predicate' );
    window.assert && assert.throws( function() { e.getUniqueTrail( function( node ) { return false; } ); }, 'Fails on false predicate' );
    assert.ok( e.getUniqueTrail( function( node ) { return node === a; } ).equals( new Trail( [ a, c, e ] ) ) );
    assert.ok( e.getUniqueTrail( function( node ) { return node === c; } ).equals( new Trail( [ c, e ] ) ) );
    assert.ok( e.getUniqueTrail( function( node ) { return node === e; } ).equals( new Trail( [ e ] ) ) );
    assert.ok( d.getUniqueTrail( function( node ) { return node === b; } ).equals( new Trail( [ b, d ] ) ) );
    assert.ok( d.getUniqueTrail( function( node ) { return node === c; } ).equals( new Trail( [ c, d ] ) ) );
    assert.ok( d.getUniqueTrail( function( node ) { return node === d; } ).equals( new Trail( [ d ] ) ) );

    // getTrails( predicate )
    trails = d.getTrails( function( node ) { return false; } );
    assert.ok( trails.length === 0 );
    trails = d.getTrails( function( node ) { return true; } );
    assert.ok( compareTrailArrays( trails, [
      new Trail( [ a, b, d ] ),
      new Trail( [ b, d ] ),
      new Trail( [ a, c, d ] ),
      new Trail( [ c, d ] ),
      new Trail( [ d ] )
    ] ) );
    trails = d.getTrails( function( node ) { return node === a; } );
    assert.ok( compareTrailArrays( trails, [
      new Trail( [ a, b, d ] ),
      new Trail( [ a, c, d ] )
    ] ) );
    trails = d.getTrails( function( node ) { return node === b; } );
    assert.ok( compareTrailArrays( trails, [
      new Trail( [ b, d ] )
    ] ) );
    trails = d.getTrails( function( node ) { return node.parents.length === 1; } );
    assert.ok( compareTrailArrays( trails, [
      new Trail( [ b, d ] ),
      new Trail( [ c, d ] )
    ] ) );
  } );

  QUnit.test( 'getLeafTrails', function( assert ) {
    var a = new Node();
    var b = new Node();
    var c = new Node();
    var d = new Node();
    var e = new Node();

    // DAG-like structure
    a.addChild( b );
    a.addChild( c );
    b.addChild( d );
    c.addChild( d );
    c.addChild( e );

    // getUniqueLeafTrail()
    window.assert && assert.throws( function() { a.getUniqueLeafTrail(); }, 'A has no unique leaf trail, since there are three' );
    assert.ok( b.getUniqueLeafTrail().equals( new Trail( [ b, d ] ) ), 'a.getUniqueLeafTrail()' );
    assert.ok( d.getUniqueLeafTrail().equals( new Trail( [ d ] ) ), 'b.getUniqueLeafTrail()' );
    assert.ok( e.getUniqueLeafTrail().equals( new Trail( [ e ] ) ), 'c.getUniqueLeafTrail()' );

    // getLeafTrails()
    var trails;
    trails = a.getLeafTrails();
    assert.ok( trails.length === 3 && compareTrailArrays( trails, [
      new Trail( [ a, b, d ] ),
      new Trail( [ a, c, d ] ),
      new Trail( [ a, c, e ] )
    ] ), 'a.getLeafTrails()' );
    trails = b.getLeafTrails();
    assert.ok( trails.length === 1 && trails[ 0 ].equals( new Trail( [ b, d ] ) ), 'b.getLeafTrails()' );
    trails = c.getLeafTrails();
    assert.ok( trails.length === 2 && compareTrailArrays( trails, [
      new Trail( [ c, d ] ),
      new Trail( [ c, e ] )
    ] ), 'c.getLeafTrails()' );
    trails = d.getLeafTrails();
    assert.ok( trails.length === 1 && trails[ 0 ].equals( new Trail( [ d ] ) ), 'd.getLeafTrails()' );
    trails = e.getLeafTrails();
    assert.ok( trails.length === 1 && trails[ 0 ].equals( new Trail( [ e ] ) ), 'e.getLeafTrails()' );

    // getUniqueLeafTrail( predicate )
    window.assert && assert.throws( function() { e.getUniqueLeafTrail( function( node ) { return false; } ); }, 'Fails on false predicate' );
    window.assert && assert.throws( function() { a.getUniqueLeafTrail( function( node ) { return true; } ); }, 'Fails on multiples' );
    assert.ok( a.getUniqueLeafTrail( function( node ) { return node === e; } ).equals( new Trail( [ a, c, e ] ) ) );

    // getLeafTrails( predicate )
    trails = a.getLeafTrails( function( node ) { return false; } );
    assert.ok( trails.length === 0 );
    trails = a.getLeafTrails( function( node ) { return true; } );
    assert.ok( compareTrailArrays( trails, [
      new Trail( [ a ] ),
      new Trail( [ a, b ] ),
      new Trail( [ a, b, d ] ),
      new Trail( [ a, c ] ),
      new Trail( [ a, c, d ] ),
      new Trail( [ a, c, e ] )
    ] ) );

    // getLeafTrailsTo( node )
    trails = a.getLeafTrailsTo( d );
    assert.ok( compareTrailArrays( trails, [
      new Trail( [ a, b, d ] ),
      new Trail( [ a, c, d ] )
    ] ) );
  } );

  QUnit.test( 'Line stroked bounds', function( assert ) {
    var line = new Line( 0, 0, 50, 0, { stroke: 'red', lineWidth: 5 } );

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

        assert.ok( line.bounds.equalsEpsilon( line.getShape().getStrokedShape( line.getLineStyles() ).bounds, 0.0001 ),
          'Line stroked bounds with ' + JSON.stringify( position ) + ' and ' + cap + ' ' + line.bounds.toString() );
      } );
    } );
  } );

  QUnit.test( 'maxWidth/maxHeight for Node', function( assert ) {
    var rect = new Rectangle( 0, 0, 100, 50, { fill: 'red' } );
    var node = new Node( { children: [ rect ] } );

    assert.ok( node.bounds.equals( new Bounds2( 0, 0, 100, 50 ) ), 'Initial bounds' );

    node.maxWidth = 50;

    assert.ok( node.bounds.equals( new Bounds2( 0, 0, 50, 25 ) ), 'Halved transform after max width of half' );

    node.maxWidth = 120;

    assert.ok( node.bounds.equals( new Bounds2( 0, 0, 100, 50 ) ), 'Back to normal after a big max width' );

    node.scale( 2 );

    assert.ok( node.bounds.equals( new Bounds2( 0, 0, 200, 100 ) ), 'Scale up should be unaffected' );

    node.maxWidth = 25;

    assert.ok( node.bounds.equals( new Bounds2( 0, 0, 50, 25 ) ), 'Scaled back down with both applied' );

    node.maxWidth = null;

    assert.ok( node.bounds.equals( new Bounds2( 0, 0, 200, 100 ) ), 'Without maxWidth' );

    node.scale( 0.5 );

    assert.ok( node.bounds.equals( new Bounds2( 0, 0, 100, 50 ) ), 'Back to normal' );

    node.left = 50;

    assert.ok( node.bounds.equals( new Bounds2( 50, 0, 150, 50 ) ), 'After a translation' );

    node.maxWidth = 50;

    assert.ok( node.bounds.equals( new Bounds2( 50, 0, 100, 25 ) ), 'maxWidth being applied after a translation, in local frame' );

    rect.rectWidth = 200;

    assert.ok( node.bounds.equals( new Bounds2( 50, 0, 100, 12.5 ) ), 'Now with a bigger rectangle' );

    rect.rectWidth = 100;
    node.maxWidth = null;

    assert.ok( node.bounds.equals( new Bounds2( 50, 0, 150, 50 ) ), 'Back to a translation' );

    rect.maxWidth = 50;

    assert.ok( node.bounds.equals( new Bounds2( 50, 0, 100, 25 ) ), 'After maxWidth A' );

    rect.maxHeight = 12.5;

    assert.ok( node.bounds.equals( new Bounds2( 50, 0, 75, 12.5 ) ), 'After maxHeight A' );
  } );

  QUnit.test( 'Spacers', function( assert ) {
    var spacer = new Spacer( 100, 50, { x: 50 } );
    assert.ok( spacer.bounds.equals( new Bounds2( 50, 0, 150, 50 ) ), 'Spacer bounds with translation' );

    var hstrut = new HStrut( 100, { y: 50 } );
    assert.ok( hstrut.bounds.equals( new Bounds2( 0, 50, 100, 50 ) ), 'HStrut bounds with translation' );

    var vstrut = new VStrut( 100, { x: 50 } );
    assert.ok( vstrut.bounds.equals( new Bounds2( 50, 0, 50, 100 ) ), 'VStrut bounds with translation' );

    assert.throws( function() {
      spacer.addChild( new Node() );
    }, 'No way to add children to Spacer' );

    assert.throws( function() {
      hstrut.addChild( new Node() );
    }, 'No way to add children to HStrut' );

    assert.throws( function() {
      vstrut.addChild( new Node() );
    }, 'No way to add children to VStrut' );
  } );

  QUnit.test( 'Renderer Summary', function( assert ) {
    var canvasNode = new CanvasNode( { canvasBounds: new Bounds2( 0, 0, 10, 10 ) } );
    var webglNode = new WebGLNode( function() {}, { canvasBounds: new Bounds2( 0, 0, 10, 10 ) } );
    var rect = new Rectangle( 0, 0, 100, 50 );
    var node = new Node( { children: [ canvasNode, webglNode, rect ] } );
    var emptyNode = new Node();

    assert.ok( canvasNode._rendererSummary.isSubtreeFullyCompatible( Renderer.bitmaskCanvas ), 'CanvasNode fully compatible: Canvas' );
    assert.ok( !canvasNode._rendererSummary.isSubtreeFullyCompatible( Renderer.bitmaskSVG ), 'CanvasNode not fully compatible: SVG' );
    assert.ok( canvasNode._rendererSummary.isSubtreeContainingCompatible( Renderer.bitmaskCanvas ), 'CanvasNode partially compatible: Canvas' );
    assert.ok( !canvasNode._rendererSummary.isSubtreeContainingCompatible( Renderer.bitmaskSVG ), 'CanvasNode not partially compatible: SVG' );
    assert.ok( canvasNode._rendererSummary.isSingleCanvasSupported(), 'CanvasNode supports single Canvas' );
    assert.ok( !canvasNode._rendererSummary.isSingleSVGSupported(), 'CanvasNode does not support single SVG' );
    assert.ok( !canvasNode._rendererSummary.isNotPainted(), 'CanvasNode is painted' );
    assert.ok( canvasNode._rendererSummary.areBoundsValid(), 'CanvasNode has valid bounds' );

    assert.ok( webglNode._rendererSummary.isSubtreeFullyCompatible( Renderer.bitmaskWebGL ), 'WebGLNode fully compatible: WebGL' );
    assert.ok( !webglNode._rendererSummary.isSubtreeFullyCompatible( Renderer.bitmaskSVG ), 'WebGLNode not fully compatible: SVG' );
    assert.ok( webglNode._rendererSummary.isSubtreeContainingCompatible( Renderer.bitmaskWebGL ), 'WebGLNode partially compatible: WebGL' );
    assert.ok( !webglNode._rendererSummary.isSubtreeContainingCompatible( Renderer.bitmaskSVG ), 'WebGLNode not partially compatible: SVG' );
    assert.ok( !webglNode._rendererSummary.isSingleCanvasSupported(), 'WebGLNode does not support single Canvas' );
    assert.ok( !webglNode._rendererSummary.isSingleSVGSupported(), 'WebGLNode does not support single SVG' );
    assert.ok( !webglNode._rendererSummary.isNotPainted(), 'WebGLNode is painted' );
    assert.ok( webglNode._rendererSummary.areBoundsValid(), 'WebGLNode has valid bounds' );

    assert.ok( rect._rendererSummary.isSubtreeFullyCompatible( Renderer.bitmaskCanvas ), 'Rectangle fully compatible: Canvas' );
    assert.ok( rect._rendererSummary.isSubtreeFullyCompatible( Renderer.bitmaskSVG ), 'Rectangle fully compatible: SVG' );
    assert.ok( rect._rendererSummary.isSubtreeContainingCompatible( Renderer.bitmaskCanvas ), 'Rectangle partially compatible: Canvas' );
    assert.ok( rect._rendererSummary.isSubtreeContainingCompatible( Renderer.bitmaskSVG ), 'Rectangle partially compatible: SVG' );
    assert.ok( rect._rendererSummary.isSingleCanvasSupported(), 'Rectangle does support single Canvas' );
    assert.ok( rect._rendererSummary.isSingleSVGSupported(), 'Rectangle does support single SVG' );
    assert.ok( !rect._rendererSummary.isNotPainted(), 'Rectangle is painted' );
    assert.ok( rect._rendererSummary.areBoundsValid(), 'Rectangle has valid bounds' );

    assert.ok( !node._rendererSummary.isSubtreeFullyCompatible( Renderer.bitmaskCanvas ), 'Container node fully compatible: Canvas' );
    assert.ok( !node._rendererSummary.isSubtreeFullyCompatible( Renderer.bitmaskSVG ), 'Container node not fully compatible: SVG' );
    assert.ok( node._rendererSummary.isSubtreeContainingCompatible( Renderer.bitmaskCanvas ), 'Container node partially compatible: Canvas' );
    assert.ok( node._rendererSummary.isSubtreeContainingCompatible( Renderer.bitmaskSVG ), 'Container node partially compatible: SVG' );
    assert.ok( !node._rendererSummary.isSingleCanvasSupported(), 'Container node does not support single Canvas' );
    assert.ok( !node._rendererSummary.isSingleSVGSupported(), 'Container node does not support single SVG' );
    assert.ok( !node._rendererSummary.isNotPainted(), 'Container node is painted' );
    assert.ok( node._rendererSummary.areBoundsValid(), 'Container node has valid bounds' );

    assert.ok( emptyNode._rendererSummary.isSubtreeFullyCompatible( Renderer.bitmaskCanvas ), 'Empty node fully compatible: Canvas' );
    assert.ok( emptyNode._rendererSummary.isSubtreeFullyCompatible( Renderer.bitmaskSVG ), 'Empty node fully compatible: SVG' );
    assert.ok( !emptyNode._rendererSummary.isSubtreeContainingCompatible( Renderer.bitmaskCanvas ), 'Empty node partially compatible: Canvas' );
    assert.ok( !emptyNode._rendererSummary.isSubtreeContainingCompatible( Renderer.bitmaskSVG ), 'Empty node partially compatible: SVG' );
    assert.ok( emptyNode._rendererSummary.isSingleCanvasSupported(), 'Empty node supports single Canvas' );
    assert.ok( emptyNode._rendererSummary.isSingleSVGSupported(), 'Empty node supports single SVG' );
    assert.ok( emptyNode._rendererSummary.isNotPainted(), 'Empty node is not painted' );
    assert.ok( emptyNode._rendererSummary.areBoundsValid(), 'Empty node has valid bounds' );
  } );

} );