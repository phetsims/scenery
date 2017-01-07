// Copyright 2002-2014, University of Colorado Boulder

/**
 * Unit tests dealing with Scenery's Display or general graphical display tests.
 */

(function() {
  'use strict';

  module( 'Scenery: Display' );


  test( 'Drawables (Rectangle)', function() {

    // The stubDisplay It's a hack that implements the subset of the Display API needed where called. It will definitely
    // be removed. The reason it stores the frame ID is because much of Scenery 0.2 uses ID comparison to determine
    // dirty state. That allows us to not have to set dirty states back to "clean" afterwards.  See #296
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new scenery.CanvasContextWrapper( canvas, context );


    var r1 = new scenery.Rectangle( 5, 10, 100, 50, 0, 0, { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new scenery.Instance( stubDisplay, r1.getUniqueTrail() );
    var r1dd = r1.createDOMDrawable( scenery.Renderer.bitmaskDOM, r1i );
    var r1ds = r1.createSVGDrawable( scenery.Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( scenery.Renderer.bitmaskCanvas, r1i );

    ok( r1._drawables.length === 3, 'After init, should have drawable refs' );

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

    ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    ok( scenery.RectangleDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.RectangleSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.RectangleCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  test( 'Drawables (Circle)', function() {
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new scenery.CanvasContextWrapper( canvas, context );


    var r1 = new scenery.Circle( 50, { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new scenery.Instance( stubDisplay, r1.getUniqueTrail() );
    var r1dd = r1.createDOMDrawable( scenery.Renderer.bitmaskDOM, r1i );
    var r1ds = r1.createSVGDrawable( scenery.Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( scenery.Renderer.bitmaskCanvas, r1i );

    ok( r1._drawables.length === 3, 'After init, should have drawable refs' );

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

    ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    ok( scenery.CircleDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.CircleSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.CircleCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  test( 'Drawables (Line)', function() {
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new scenery.CanvasContextWrapper( canvas, context );


    var r1 = new scenery.Line( 0, 1, 2, 3, { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new scenery.Instance( stubDisplay, r1.getUniqueTrail() );
    var r1ds = r1.createSVGDrawable( scenery.Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( scenery.Renderer.bitmaskCanvas, r1i );

    ok( r1._drawables.length === 2, 'After init, should have drawable refs' );

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

    ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    ok( scenery.LineSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.LineCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  test( 'Drawables (Path)', function() {
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new scenery.CanvasContextWrapper( canvas, context );


    var r1 = new scenery.Path( kite.Shape.regularPolygon( 5, 10 ), { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new scenery.Instance( stubDisplay, r1.getUniqueTrail() );
    var r1ds = r1.createSVGDrawable( scenery.Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( scenery.Renderer.bitmaskCanvas, r1i );

    ok( r1._drawables.length === 2, 'After init, should have drawable refs' );

    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );

    r1.shape = kite.Shape.regularPolygon( 6, 20 );

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

    ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    ok( scenery.PathSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.PathCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  test( 'Drawables (Text)', function() {
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new scenery.CanvasContextWrapper( canvas, context );


    var r1 = new scenery.Text( 'Wow!', { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new scenery.Instance( stubDisplay, r1.getUniqueTrail() );
    var r1dd = r1.createDOMDrawable( scenery.Renderer.bitmaskDOM, r1i );
    var r1ds = r1.createSVGDrawable( scenery.Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( scenery.Renderer.bitmaskCanvas, r1i );

    ok( r1._drawables.length === 3, 'After init, should have drawable refs' );

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

    ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    ok( scenery.TextDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.TextSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.TextCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  test( 'Drawables (Image)', function() {
    var stubDisplay = { _frameId: 5 };

    var canvas = document.createElement( 'canvas' );
    canvas.width = 64;
    canvas.height = 48;
    var context = canvas.getContext( '2d' );
    var wrapper = new scenery.CanvasContextWrapper( canvas, context );

    // 1x1 black PNG
    var r1 = new scenery.Image( 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIW2NkYGD4DwABCQEBtxmN7wAAAABJRU5ErkJggg==' );
    var r1i = new scenery.Instance( stubDisplay, r1.getUniqueTrail() );
    var r1dd = r1.createDOMDrawable( scenery.Renderer.bitmaskDOM, r1i );
    var r1ds = r1.createSVGDrawable( scenery.Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( scenery.Renderer.bitmaskCanvas, r1i );

    ok( r1._drawables.length === 3, 'After init, should have drawable refs' );

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

    ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    ok( scenery.ImageDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.ImageSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.ImageCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  test( 'Drawables (DOM)', function() {
    var stubDisplay = { _frameId: 5 };

    var r1 = new scenery.DOM( document.createElement( 'canvas' ) );
    var r1i = new scenery.Instance( stubDisplay, r1.getUniqueTrail() );
    var r1dd = r1.createDOMDrawable( scenery.Renderer.bitmaskDOM, r1i );

    ok( r1._drawables.length === 1, 'After init, should have drawable refs' );

    r1dd.updateDOM();

    r1dd.dispose();

    ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

    ok( scenery.DOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );

  test( 'Renderer order bitmask', function() {
    var Renderer = scenery.Renderer;

    // init test
    var mask = Renderer.createOrderBitmask( Renderer.bitmaskCanvas, Renderer.bitmaskSVG, Renderer.bitmaskDOM, Renderer.bitmaskWebGL );
    equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrder( mask, 3 ), Renderer.bitmaskWebGL );

    // empty test
    mask = Renderer.createOrderBitmask();
    equal( Renderer.bitmaskOrder( mask, 0 ), 0 );
    equal( Renderer.bitmaskOrder( mask, 1 ), 0 );
    equal( Renderer.bitmaskOrder( mask, 2 ), 0 );
    equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing single renderer should work
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 1 ), 0 );
    equal( Renderer.bitmaskOrder( mask, 2 ), 0 );
    equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing again should have no change
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 1 ), 0 );
    equal( Renderer.bitmaskOrder( mask, 2 ), 0 );
    equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing Canvas will put it first, SVG second
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 2 ), 0 );
    equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing SVG will reverse the two
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrder( mask, 2 ), 0 );
    equal( Renderer.bitmaskOrder( mask, 3 ), 0 );
    equal( Renderer.bitmaskOrder( mask, 4 ), 0 );

    // pushing DOM shifts the other two down
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing DOM results in no change
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing Canvas moves it to the front
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 3 ), 0 );

    // pushing DOM again swaps it with the Canvas
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrder( mask, 3 ), 0 );
    // console.log( mask.toString( 16 ) );
    // pushing WebGL shifts everything
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskWebGL );
    equal( Renderer.bitmaskOrder( mask, 0 ), Renderer.bitmaskWebGL );
    equal( Renderer.bitmaskOrder( mask, 1 ), Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrder( mask, 2 ), Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrder( mask, 3 ), Renderer.bitmaskSVG );
    // console.log( mask.toString( 16 ) );
  } );

  /* eslint-disable no-undef */

  test( 'Empty Display usage', function() {
    var n = new scenery.Node();
    var d = new scenery.Display( n );
    d.updateDisplay();
    d.updateDisplay();

    expect( 0 );
  } );

  test( 'Simple Display usage', function() {
    var r = new scenery.Rectangle( 0, 0, 50, 50, { fill: 'red' } );
    var d = new scenery.Display( r );
    d.updateDisplay();
    r.rectWidth = 100;
    d.updateDisplay();

    expect( 0 );
  } );

  test( 'Stitch patterns #1', function() {
    var n = new scenery.Node();
    var d = new scenery.Display( n );
    d.updateDisplay();

    n.addChild( new scenery.Rectangle( 0, 0, 50, 50, { fill: 'red' } ) );
    d.updateDisplay();

    n.addChild( new scenery.Rectangle( 0, 0, 50, 50, { fill: 'red' } ) );
    d.updateDisplay();

    n.addChild( new scenery.Rectangle( 0, 0, 50, 50, { fill: 'red' } ) );
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

    expect( 0 );
  } );

  test( 'Invisible append', function() {
    var scene = new scenery.Node();
    var display = new scenery.Display( scene );
    display.updateDisplay();

    var a = new scenery.Rectangle( 0, 0, 100, 50, { fill: 'red' } );
    scene.addChild( a );
    display.updateDisplay();

    var b = new scenery.Rectangle( 0, 0, 100, 50, { fill: 'red', visible: false } );
    scene.addChild( b );
    display.updateDisplay();

    expect( 0 );
  } );

  test( 'Stitching problem A (GitHub Issue #339)', function() {
    var scene = new scenery.Node();
    var display = new scenery.Display( scene );

    var a = new scenery.Rectangle( 0, 0, 100, 50, { fill: 'red' } );
    var b = new scenery.Rectangle( 0, 0, 50, 50, { fill: 'blue' } );
    var c = new scenery.DOM( document.createElement( 'div' ) );
    var d = new scenery.Rectangle( 100, 0, 100, 50, { fill: 'red' } );
    var e = new scenery.Rectangle( 100, 0, 50, 50, { fill: 'blue' } );

    var f = new scenery.Rectangle( 0, 50, 100, 50, { fill: 'green' } );
    var g = new scenery.DOM( document.createElement( 'div' ) );

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

    expect( 0 );
  } );

  test( 'SVG group disposal issue (GitHub Issue #354) A', function() {
    var scene = new scenery.Node();
    var display = new scenery.Display( scene );

    var node = new scenery.Node( {
      renderer: 'svg',
      cssTransform: true
    } );
    var rect = new scenery.Rectangle( 0, 0, 100, 50, { fill: 'red' } );

    scene.addChild( node );
    node.addChild( rect );
    display.updateDisplay();

    scene.removeChild( node );
    display.updateDisplay();

    expect( 0 );
  } );

  test( 'SVG group disposal issue (GitHub Issue #354) B', function() {
    var scene = new scenery.Node();
    var display = new scenery.Display( scene );

    var node = new scenery.Node();
    var rect = new scenery.Rectangle( 0, 0, 100, 50, {
      fill: 'red',
      renderer: 'svg',
      cssTransform: true
    } );

    scene.addChild( node );
    node.addChild( rect );
    display.updateDisplay();

    scene.removeChild( node );
    display.updateDisplay();

    expect( 0 );
  } );

  test( 'Empty path display test', function() {
    var scene = new scenery.Node();
    var display = new scenery.Display( scene );

    scene.addChild( new scenery.Path( null ) );
    display.updateDisplay();

    expect( 0 );
  } );

  test( 'Double remove related to #392', function() {
    var scene = new scenery.Node();
    var display = new scenery.Display( scene );

    display.updateDisplay();

    var n1 = new scenery.Node();
    var n2 = new scenery.Node();
    scene.addChild( n1 );
    n1.addChild( n2 );
    scene.addChild( n2 ); // so the tree has a reference to the Node that we can trigger the failure on

    display.updateDisplay();

    scene.removeChild( n1 );
    n1.removeChild( n2 );

    display.updateDisplay();

    expect( 0 );
  } );

  /* eslint-enable */

})();
