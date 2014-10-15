//  Copyright 2002-2014, University of Colorado Boulder

(function() {
  'use strict';

  module( 'Scenery: Instances' );

  // Borrowed from tests/qunit/js/display.js
  test( 'Instances (RenderState)', function() {
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

    ok( r1i.hasOwnProperty( 'renderState' ), 'After Rendering, the instance should have a defined renderState' );
    ok( r1i.renderState !== null, 'After Rendering, the instance should have a non-null renderState' );

    var initialRenderState = r1i.renderState;

    r1.setFill( 'green' );

    ok( r1i.renderState === initialRenderState, 'RenderState instances should be mutable and reused, see #292' );

    r1dd.dispose();
    r1ds.dispose();
    r1dc.dispose();
  } )
})();
