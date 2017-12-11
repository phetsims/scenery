// Copyright 2017, University of Colorado Boulder

/**
 * Display tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var CanvasContextWrapper = require( 'SCENERY/util/CanvasContextWrapper' );
  var Circle = require( 'SCENERY/nodes/Circle' );
  var CircleCanvasDrawable = require( 'SCENERY/display/drawables/CircleCanvasDrawable' );
  var CircleDOMDrawable = require( 'SCENERY/display/drawables/CircleDOMDrawable' );
  var CircleSVGDrawable = require( 'SCENERY/display/drawables/CircleSVGDrawable' );
  var Display = require( 'SCENERY/display/Display' );
  var DOM = require( 'SCENERY/nodes/DOM' );
  var DOMDrawable = require( 'SCENERY/display/drawables/DOMDrawable' );
  var Image = require( 'SCENERY/nodes/Image' );
  var ImageCanvasDrawable = require( 'SCENERY/display/drawables/ImageCanvasDrawable' );
  var ImageDOMDrawable = require( 'SCENERY/display/drawables/ImageDOMDrawable' );
  var ImageSVGDrawable = require( 'SCENERY/display/drawables/ImageSVGDrawable' );
  var Instance = require( 'SCENERY/display/Instance' );
  var Line = require( 'SCENERY/nodes/Line' );
  var LineCanvasDrawable = require( 'SCENERY/display/drawables/LineCanvasDrawable' );
  var LineSVGDrawable = require( 'SCENERY/display/drawables/LineSVGDrawable' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Path = require( 'SCENERY/nodes/Path' );
  var PathCanvasDrawable = require( 'SCENERY/display/drawables/PathCanvasDrawable' );
  var PathSVGDrawable = require( 'SCENERY/display/drawables/PathSVGDrawable' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var RectangleCanvasDrawable = require( 'SCENERY/display/drawables/RectangleCanvasDrawable' );
  var RectangleDOMDrawable = require( 'SCENERY/display/drawables/RectangleDOMDrawable' );
  var RectangleSVGDrawable = require( 'SCENERY/display/drawables/RectangleSVGDrawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var Shape = require( 'KITE/Shape' );
  var Text = require( 'SCENERY/nodes/Text' );
  var TextCanvasDrawable = require( 'SCENERY/display/drawables/TextCanvasDrawable' );
  var TextDOMDrawable = require( 'SCENERY/display/drawables/TextDOMDrawable' );
  var TextSVGDrawable = require( 'SCENERY/display/drawables/TextSVGDrawable' );

  QUnit.module( 'Display' );

  QUnit.test( 'Drawables (Rectangle)', function( assert ) {

    // The stubDisplay It's a hack that implements the subset of the Display API needed where called. It will definitely
    // be removed. The reason it stores the frame ID is because much of Scenery 0.2 uses ID comparison to determine
    // dirty state. That allows us to not have to set dirty states back to "clean" afterwards.  See #296
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new CanvasContextWrapper( canvas, context );


    var r1 = new Rectangle( 5, 10, 100, 50, 0, 0, { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
    var r1dd = r1.createDOMDrawable( Renderer.bitmaskDOM, r1i );
    var r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

    assert.ok( r1._drawables.length === 3, 'After init, should have drawable refs' );

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.setRect( 0, 0, 100, 100, 5, 5 );

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.stroke = null;
    r1.fill = null;

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1dd.dispose();
    r1ds.dispose();
    r1dc.dispose();

    assert.ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    assert.ok( RectangleDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    assert.ok( RectangleSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    assert.ok( RectangleCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  QUnit.test( 'Drawables (Circle)', function( assert ) {
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new CanvasContextWrapper( canvas, context );


    var r1 = new Circle( 50, { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
    var r1dd = r1.createDOMDrawable( Renderer.bitmaskDOM, r1i );
    var r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

    assert.ok( r1._drawables.length === 3, 'After init, should have drawable refs' );

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.setRadius( 100 );

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.stroke = null;
    r1.fill = null;

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1dd.dispose();
    r1ds.dispose();
    r1dc.dispose();

    assert.ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    assert.ok( CircleDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    assert.ok( CircleSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    assert.ok( CircleCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  QUnit.test( 'Drawables (Line)', function( assert ) {
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new CanvasContextWrapper( canvas, context );

    var r1 = new Line( 0, 1, 2, 3, { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
    var r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

    assert.ok( r1._drawables.length === 2, 'After init, should have drawable refs' );

    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.x1 = 50;
    r1.x2 = 100;

    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.stroke = null;
    r1.fill = null;

    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1ds.dispose();
    r1dc.dispose();

    assert.ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    assert.ok( LineSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    assert.ok( LineCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  QUnit.test( 'Drawables (Path)', function( assert ) {
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new CanvasContextWrapper( canvas, context );


    var r1 = new Path( Shape.regularPolygon( 5, 10 ), { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
    var r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

    assert.ok( r1._drawables.length === 2, 'After init, should have drawable refs' );

    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.shape = Shape.regularPolygon( 6, 20 );

    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.stroke = null;
    r1.fill = null;

    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.shape = null;

    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1ds.dispose();
    r1dc.dispose();

    assert.ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    assert.ok( PathSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    assert.ok( PathCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  QUnit.test( 'Drawables (Text)', function( assert ) {
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new CanvasContextWrapper( canvas, context );


    var r1 = new Text( 'Wow!', { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
    var r1dd = r1.createDOMDrawable( Renderer.bitmaskDOM, r1i );
    var r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

    assert.ok( r1._drawables.length === 3, 'After init, should have drawable refs' );

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.text = 'b';

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.font = '20px sans-serif';

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.stroke = null;
    r1.fill = null;

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1dd.dispose();
    r1ds.dispose();
    r1dc.dispose();

    assert.ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    assert.ok( TextDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    assert.ok( TextSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    assert.ok( TextCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  QUnit.test( 'Drawables (Image)', function( assert ) {
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new CanvasContextWrapper( canvas, context );

    // 1x1 black PNG
    var r1 = new Image( 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIW2NkYGD4DwABCQEBtxmN7wAAAABJRU5ErkJggg==' );
    var r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
    var r1dd = r1.createDOMDrawable( Renderer.bitmaskDOM, r1i );
    var r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

    assert.ok( r1._drawables.length === 3, 'After init, should have drawable refs' );

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    // 1x1 black JPG
    r1.image = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD8qqKKKAP/2Q==';

    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1dd.dispose();
    r1ds.dispose();
    r1dc.dispose();

    assert.ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    assert.ok( ImageDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    assert.ok( ImageSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    assert.ok( ImageCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  QUnit.test( 'Drawables (DOM)', function( assert ) {
    var stubDisplay = { _frameId: 5 };

    var r1 = new DOM( document.createElement( 'canvas' ) );
    var r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
    var r1dd = r1.createDOMDrawable( Renderer.bitmaskDOM, r1i );

    assert.ok( r1._drawables.length === 1, 'After init, should have drawable refs' );

    r1dd.updateDOM();

    r1dd.dispose();

    assert.ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    assert.ok( DOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  QUnit.test( 'Renderer order bitmask', function( assert ) {

    // init test
    var mask = Renderer.createOrderBitmask( Renderer.bitmaskCanvas, Renderer.bitmaskSVG, Renderer.bitmaskDOM, Renderer.bitmaskWebGL );
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskCanvas );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskDOM );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), Renderer.bitmaskWebGL );

    // empty test
    mask = Renderer.createOrderBitmask();
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), 0 );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), 0 );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), 0 );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing single renderer should work
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), 0 );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), 0 );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing again should have no change
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), 0 );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), 0 );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing Canvas will put it first, SVG second
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskCanvas );
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskCanvas );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), 0 );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing SVG will reverse the two
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskCanvas );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), 0 );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), 0 );
    assert.equal( Renderer.bitmaskOrder( mask, 4 ), 0 );

    // pushing DOM shifts the other two down
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskDOM );
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskDOM );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskCanvas );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing DOM results in no change
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskDOM );
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskDOM );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskCanvas );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing Canvas moves it to the front
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskCanvas );
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskCanvas );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskDOM );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing DOM again swaps it with the Canvas
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskDOM );
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskDOM );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskCanvas );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskSVG );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), 0 );
    // console.log( mask.toString( 16 ) );
    // pushing WebGL shifts everything
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskWebGL );
    assert.equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskWebGL );
    assert.equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskDOM );
    assert.equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskCanvas );
    assert.equal( Renderer.bitmaskOrder( mask, 3 ), Renderer.bitmaskSVG );
    // console.log( mask.toString( 16 ) );
  } );

  /* eslint-disable no-undef */

  QUnit.test( 'Empty Display usage', function( assert ) {
    var n = new Node();
    var d = new Display( n );
    d.updateDisplay();
    d.updateDisplay();

    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'Simple Display usage', function( assert ) {
    var r = new Rectangle( 0, 0, 50, 50, { fill: 'red' } );
    var d = new Display( r );
    d.updateDisplay();
    r.rectWidth = 100;
    d.updateDisplay();

    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'Stitch patterns #1', function( assert ) {
    var n = new Node();
    var d = new Display( n );
    d.updateDisplay();

    n.addChild( new Rectangle( 0, 0, 50, 50, { fill: 'red' } ) );
    d.updateDisplay();

    n.addChild( new Rectangle( 0, 0, 50, 50, { fill: 'red' } ) );
    d.updateDisplay();

    n.addChild( new Rectangle( 0, 0, 50, 50, { fill: 'red' } ) );
    d.updateDisplay();

    n.children[ 1 ].visible = false;
    d.updateDisplay();

    n.children[ 1 ].visible = true;
    d.updateDisplay();

    n.removeChild( n.children[ 0 ] );
    d.updateDisplay();

    n.removeChild( n.children[ 1 ] );
    d.updateDisplay();

    n.removeChild( n.children[ 0 ] );
    d.updateDisplay();

    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'Invisible append', function( assert ) {
    var scene = new Node();
    var display = new Display( scene );
    display.updateDisplay();

    var a = new Rectangle( 0, 0, 100, 50, { fill: 'red' } );
    scene.addChild( a );
    display.updateDisplay();

    var b = new Rectangle( 0, 0, 100, 50, { fill: 'red', visible: false } );
    scene.addChild( b );
    display.updateDisplay();

    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'Stitching problem A (GitHub Issue #339)', function( assert ) {
    var scene = new Node();
    var display = new Display( scene );

    var a = new Rectangle( 0, 0, 100, 50, { fill: 'red' } );
    var b = new Rectangle( 0, 0, 50, 50, { fill: 'blue' } );
    var c = new DOM( document.createElement( 'div' ) );
    var d = new Rectangle( 100, 0, 100, 50, { fill: 'red' } );
    var e = new Rectangle( 100, 0, 50, 50, { fill: 'blue' } );

    var f = new Rectangle( 0, 50, 100, 50, { fill: 'green' } );
    var g = new DOM( document.createElement( 'div' ) );

    scene.addChild( a );
    scene.addChild( f );
    scene.addChild( b );
    scene.addChild( c );
    scene.addChild( d );
    scene.addChild( e );
    display.updateDisplay();

    scene.removeChild( f );
    scene.insertChild( 4, g );
    display.updateDisplay();

    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'SVG group disposal issue (GitHub Issue #354) A', function( assert ) {
    var scene = new Node();
    var display = new Display( scene );

    var node = new Node( {
      renderer: 'svg',
      cssTransform: true
    } );
    var rect = new Rectangle( 0, 0, 100, 50, { fill: 'red' } );

    scene.addChild( node );
    node.addChild( rect );
    display.updateDisplay();

    scene.removeChild( node );
    display.updateDisplay();

    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'SVG group disposal issue (GitHub Issue #354) B', function( assert ) {
    var scene = new Node();
    var display = new Display( scene );

    var node = new Node();
    var rect = new Rectangle( 0, 0, 100, 50, {
      fill: 'red',
      renderer: 'svg',
      cssTransform: true
    } );

    scene.addChild( node );
    node.addChild( rect );
    display.updateDisplay();

    scene.removeChild( node );
    display.updateDisplay();

    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'Empty path display test', function( assert ) {
    var scene = new Node();
    var display = new Display( scene );

    scene.addChild( new Path( null ) );
    display.updateDisplay();

    assert.ok( true, 'so we have at least 1 test in this set' );
  } );

  QUnit.test( 'Double remove related to #392', function( assert ) {
    var scene = new Node();
    var display = new Display( scene );

    display.updateDisplay();

    var n1 = new Node();
    var n2 = new Node();
    scene.addChild( n1 );
    n1.addChild( n2 );
    scene.addChild( n2 ); // so the tree has a reference to the Node that we can trigger the failure on

    display.updateDisplay();

    scene.removeChild( n1 );
    n1.removeChild( n2 );

    display.updateDisplay();

    assert.ok( true, 'so we have at least 1 test in this set' );
  } );
} );