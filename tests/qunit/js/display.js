
(function(){
  'use strict';
  
  module( 'Scenery: Display' );
  
  test( 'Drawables (Rectangle)', function() {
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
    
    ok( scenery.Rectangle.RectangleDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Rectangle.RectangleSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Rectangle.RectangleCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
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
    
    ok( scenery.Circle.CircleDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Circle.CircleSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Circle.CircleCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
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
    
    ok( scenery.Line.LineSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Line.LineCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
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
    
    ok( scenery.Path.PathSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Path.PathCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
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
    
    ok( scenery.Text.TextDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Text.TextSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Text.TextCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
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
    
    r1.image = null;
    
    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper, r1 );
    
    r1dd.dispose();
    r1ds.dispose();
    r1dc.dispose();
    
    ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );
    
    ok( scenery.Image.ImageDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Image.ImageSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Image.ImageCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
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
    
    ok( scenery.DOM.DOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );
  
  test( 'Renderer order bitmask', function() {
    var Renderer = scenery.Renderer;
    
    // init test
    var mask = Renderer.createOrderBitmask( Renderer.bitmaskCanvas, Renderer.bitmaskSVG, Renderer.bitmaskDOM, Renderer.bitmaskWebGL );
    equal( Renderer.bitmaskOrderFirst( mask ),  Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrderSecond( mask ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderThird( mask ),  Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrderFourth( mask ), Renderer.bitmaskWebGL );
    
    // empty test
    mask = Renderer.createOrderBitmask();
    equal( Renderer.bitmaskOrderFirst( mask ),  0 );
    equal( Renderer.bitmaskOrderSecond( mask ), 0 );
    equal( Renderer.bitmaskOrderThird( mask ),  0 );
    equal( Renderer.bitmaskOrderFourth( mask ), 0 );
    
    // pushing single renderer should work
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderFirst( mask ),  Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderSecond( mask ), 0 );
    equal( Renderer.bitmaskOrderThird( mask ),  0 );
    equal( Renderer.bitmaskOrderFourth( mask ), 0 );
    
    // pushing again should have no change
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderFirst( mask ),  Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderSecond( mask ), 0 );
    equal( Renderer.bitmaskOrderThird( mask ),  0 );
    equal( Renderer.bitmaskOrderFourth( mask ), 0 );
    
    // pushing Canvas will put it first, SVG second
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrderFirst( mask ),  Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrderSecond( mask ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderThird( mask ),  0 );
    equal( Renderer.bitmaskOrderFourth( mask ), 0 );
    
    // pushing SVG will reverse the two
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderFirst( mask ),  Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderSecond( mask ), Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrderThird( mask ),  0 );
    equal( Renderer.bitmaskOrderFourth( mask ), 0 );
    
    // pushing DOM shifts the other two down
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrderFirst( mask ),  Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrderSecond( mask ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderThird( mask ),  Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrderFourth( mask ), 0 );
    
    // pushing DOM results in no change
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrderFirst( mask ),  Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrderSecond( mask ), Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderThird( mask ),  Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrderFourth( mask ), 0 );
    
    // pushing Canvas moves it to the front
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrderFirst( mask ),  Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrderSecond( mask ), Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrderThird( mask ),  Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderFourth( mask ), 0 );
    
    // pushing DOM again swaps it with the Canvas
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrderFirst( mask ),  Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrderSecond( mask ), Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrderThird( mask ),  Renderer.bitmaskSVG );
    equal( Renderer.bitmaskOrderFourth( mask ), 0 );
    console.log( mask.toString( 16 ) );
    // pushing WebGL shifts everything
    mask = Renderer.pushOrderBitmask( mask, Renderer.bitmaskWebGL );
    equal( Renderer.bitmaskOrderFirst( mask ),  Renderer.bitmaskWebGL );
    equal( Renderer.bitmaskOrderSecond( mask ), Renderer.bitmaskDOM );
    equal( Renderer.bitmaskOrderThird( mask ),  Renderer.bitmaskCanvas );
    equal( Renderer.bitmaskOrderFourth( mask ), Renderer.bitmaskSVG );
    console.log( mask.toString( 16 ) );
  } );
})();
