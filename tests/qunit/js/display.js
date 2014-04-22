
(function(){
  'use strict';
  
  module( 'Scenery: Display' );
  
  test( 'Drawables (Rectangle)', function() {
    var stubDisplay = { _frameId: 5 };
    
    var canvas = document.createElement( 'canvas' );
    var context = canvas.getContext( '2d' );
    var wrapper = new scenery.CanvasContextWrapper( canvas, context );
    
    
    var r1 = new scenery.Rectangle( 5, 10, 100, 50, 0, 0, { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new scenery.DisplayInstance( stubDisplay, r1.getUniqueTrail(), scenery.RenderState.RegularState.createRootState( r1 ) );
    var r1dd = r1.createDOMDrawable( scenery.Renderer.bitmaskDOM, r1i );
    var r1ds = r1.createSVGDrawable( scenery.Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( scenery.Renderer.bitmaskCanvas, r1i );
    
    ok( r1._drawables.length === 3, 'After init, should have drawable refs' );
    
    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper );
    
    r1.setRect( 0, 0, 100, 100, 5, 5 );
    
    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper );
    
    r1dd.dispose();
    r1ds.dispose();
    r1dc.dispose();
    
    ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );
    
    ok( scenery.Rectangle.RectangleDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Rectangle.RectangleSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Rectangle.RectangleCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );
  
  test( 'Drawables (Text)', function() {
    var stubDisplay = { _frameId: 5 };
    
    var canvas = document.createElement( 'canvas' );
    var context = canvas.getContext( '2d' );
    var wrapper = new scenery.CanvasContextWrapper( canvas, context );
    
    
    var r1 = new scenery.Text( 'Wow!', { fill: 'red', stroke: 'blue', lineWidth: 5 } );
    var r1i = new scenery.DisplayInstance( stubDisplay, r1.getUniqueTrail(), scenery.RenderState.RegularState.createRootState( r1 ) );
    var r1dd = r1.createDOMDrawable( scenery.Renderer.bitmaskDOM, r1i );
    var r1ds = r1.createSVGDrawable( scenery.Renderer.bitmaskSVG, r1i );
    var r1dc = r1.createCanvasDrawable( scenery.Renderer.bitmaskCanvas, r1i );
    
    ok( r1._drawables.length === 3, 'After init, should have drawable refs' );
    
    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper );
    
    r1.text = 'b';
    
    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper );
    
    r1.font = '20px sans-serif';
    
    r1dd.updateDOM();
    r1ds.updateSVG();
    r1dc.paintCanvas( wrapper );
    
    r1dd.dispose();
    r1ds.dispose();
    r1dc.dispose();
    
    ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );
    
    ok( scenery.Text.TextDOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Text.TextSVGDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
    ok( scenery.Text.TextCanvasDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
  } );
})();
