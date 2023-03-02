// Copyright 2017-2023, University of Colorado Boulder

/**
 * Display tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import { Shape } from '../../../kite/js/imports.js';
import Circle from '../nodes/Circle.js';
import DOM from '../nodes/DOM.js';
import Image from '../nodes/Image.js';
import Line from '../nodes/Line.js';
import Node from '../nodes/Node.js';
import Path from '../nodes/Path.js';
import Rectangle from '../nodes/Rectangle.js';
import Text from '../nodes/Text.js';
import CanvasContextWrapper from '../util/CanvasContextWrapper.js';
import Display from './Display.js';
import CircleCanvasDrawable from './drawables/CircleCanvasDrawable.js';
import CircleDOMDrawable from './drawables/CircleDOMDrawable.js';
import CircleSVGDrawable from './drawables/CircleSVGDrawable.js';
import DOMDrawable from './drawables/DOMDrawable.js';
import ImageCanvasDrawable from './drawables/ImageCanvasDrawable.js';
import ImageDOMDrawable from './drawables/ImageDOMDrawable.js';
import ImageSVGDrawable from './drawables/ImageSVGDrawable.js';
import LineCanvasDrawable from './drawables/LineCanvasDrawable.js';
import LineSVGDrawable from './drawables/LineSVGDrawable.js';
import PathCanvasDrawable from './drawables/PathCanvasDrawable.js';
import PathSVGDrawable from './drawables/PathSVGDrawable.js';
import RectangleCanvasDrawable from './drawables/RectangleCanvasDrawable.js';
import RectangleDOMDrawable from './drawables/RectangleDOMDrawable.js';
import RectangleSVGDrawable from './drawables/RectangleSVGDrawable.js';
import TextCanvasDrawable from './drawables/TextCanvasDrawable.js';
import TextDOMDrawable from './drawables/TextDOMDrawable.js';
import TextSVGDrawable from './drawables/TextSVGDrawable.js';
import Instance from './Instance.js';
import Renderer from './Renderer.js';

QUnit.module( 'Display' );

QUnit.test( 'Drawables (Rectangle)', assert => {

  // The stubDisplay It's a hack that implements the subset of the Display API needed where called. It will definitely
  // be removed. The reason it stores the frame ID is because much of Scenery 0.2 uses ID comparison to determine
  // dirty state. That allows us to not have to set dirty states back to "clean" afterwards.  See #296
  const stubDisplay = { _frameId: 5, isWebGLAllowed: () => true };

  const canvas = document.createElement( 'canvas' );
  canvas.width = 64;
  canvas.height = 48;
  const context = canvas.getContext( '2d' );
  const wrapper = new CanvasContextWrapper( canvas, context );


  const r1 = new Rectangle( 5, 10, 100, 50, 0, 0, { fill: 'red', stroke: 'blue', lineWidth: 5 } );
  const r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
  const r1dd = r1.createDOMDrawable( Renderer.bitmaskDOM, r1i );
  const r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
  const r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

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

QUnit.test( 'Drawables (Circle)', assert => {
  const stubDisplay = { _frameId: 5, isWebGLAllowed: () => true };

  const canvas = document.createElement( 'canvas' );
  canvas.width = 64;
  canvas.height = 48;
  const context = canvas.getContext( '2d' );
  const wrapper = new CanvasContextWrapper( canvas, context );


  const r1 = new Circle( 50, { fill: 'red', stroke: 'blue', lineWidth: 5 } );
  const r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
  const r1dd = r1.createDOMDrawable( Renderer.bitmaskDOM, r1i );
  const r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
  const r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

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

QUnit.test( 'Drawables (Line)', assert => {
  const stubDisplay = { _frameId: 5, isWebGLAllowed: () => true };

  const canvas = document.createElement( 'canvas' );
  canvas.width = 64;
  canvas.height = 48;
  const context = canvas.getContext( '2d' );
  const wrapper = new CanvasContextWrapper( canvas, context );

  const r1 = new Line( 0, 1, 2, 3, { fill: 'red', stroke: 'blue', lineWidth: 5 } );
  const r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
  const r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
  const r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

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

QUnit.test( 'Drawables (Path)', assert => {
  const stubDisplay = { _frameId: 5, isWebGLAllowed: () => true };

  const canvas = document.createElement( 'canvas' );
  canvas.width = 64;
  canvas.height = 48;
  const context = canvas.getContext( '2d' );
  const wrapper = new CanvasContextWrapper( canvas, context );


  const r1 = new Path( Shape.regularPolygon( 5, 10 ), { fill: 'red', stroke: 'blue', lineWidth: 5 } );
  const r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
  const r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
  const r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

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

QUnit.test( 'Drawables (Text)', assert => {
  const stubDisplay = { _frameId: 5, isWebGLAllowed: () => true };

  const canvas = document.createElement( 'canvas' );
  canvas.width = 64;
  canvas.height = 48;
  const context = canvas.getContext( '2d' );
  const wrapper = new CanvasContextWrapper( canvas, context );


  const r1 = new Text( 'Wow!', { fill: 'red', stroke: 'blue', lineWidth: 5 } );
  const r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
  const r1dd = r1.createDOMDrawable( Renderer.bitmaskDOM, r1i );
  const r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
  const r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

  assert.ok( r1._drawables.length === 3, 'After init, should have drawable refs' );

  r1dd.updateDOM();
  r1ds.updateSVG();
  r1dc.paintCanvas( wrapper, r1 );

  r1.string = 'b';

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

QUnit.test( 'Drawables (Image)', assert => {
  const stubDisplay = { _frameId: 5, isWebGLAllowed: () => true };

  const canvas = document.createElement( 'canvas' );
  canvas.width = 64;
  canvas.height = 48;
  const context = canvas.getContext( '2d' );
  const wrapper = new CanvasContextWrapper( canvas, context );

  // 1x1 black PNG
  const r1 = new Image( 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIW2NkYGD4DwABCQEBtxmN7wAAAABJRU5ErkJggg==' );
  const r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
  const r1dd = r1.createDOMDrawable( Renderer.bitmaskDOM, r1i );
  const r1ds = r1.createSVGDrawable( Renderer.bitmaskSVG, r1i );
  const r1dc = r1.createCanvasDrawable( Renderer.bitmaskCanvas, r1i );

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

QUnit.test( 'Drawables (DOM)', assert => {
  const stubDisplay = { _frameId: 5, isWebGLAllowed: () => true };

  const r1 = new DOM( document.createElement( 'canvas' ) );
  const r1i = new Instance( stubDisplay, r1.getUniqueTrail() );
  const r1dd = r1.createDOMDrawable( Renderer.bitmaskDOM, r1i );

  assert.ok( r1._drawables.length === 1, 'After init, should have drawable refs' );

  r1dd.updateDOM();

  r1dd.dispose();

  assert.ok( r1._drawables.length === 0, 'After dispose, should not have drawable refs' );

  assert.ok( DOMDrawable.pool.length > 0, 'Disposed drawable returned to pool' );
} );

QUnit.test( 'Renderer order bitmask', assert => {

  // init test
  let mask = Renderer.createOrderBitmask( Renderer.bitmaskCanvas, Renderer.bitmaskSVG, Renderer.bitmaskDOM, Renderer.bitmaskWebGL );
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
 

QUnit.test( 'Empty Display usage', assert => {
  const n = new Node();
  const d = new Display( n );
  d.updateDisplay();
  d.updateDisplay();

  assert.ok( true, 'so we have at least 1 test in this set' );
  d.dispose();
} );

QUnit.test( 'Simple Display usage', assert => {
  const r = new Rectangle( 0, 0, 50, 50, { fill: 'red' } );
  const d = new Display( r );
  d.updateDisplay();
  r.rectWidth = 100;
  d.updateDisplay();

  assert.ok( true, 'so we have at least 1 test in this set' );
  d.dispose();
} );

QUnit.test( 'Stitch patterns #1', assert => {
  const n = new Node();
  const d = new Display( n );
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
  d.dispose();
} );

QUnit.test( 'Invisible append', assert => {
  const scene = new Node();
  const display = new Display( scene );
  display.updateDisplay();

  const a = new Rectangle( 0, 0, 100, 50, { fill: 'red' } );
  scene.addChild( a );
  display.updateDisplay();

  const b = new Rectangle( 0, 0, 100, 50, { fill: 'red', visible: false } );
  scene.addChild( b );
  display.updateDisplay();

  assert.ok( true, 'so we have at least 1 test in this set' );
  display.dispose();

} );

QUnit.test( 'Stitching problem A (GitHub Issue #339)', assert => {
  const scene = new Node();
  const display = new Display( scene );

  const a = new Rectangle( 0, 0, 100, 50, { fill: 'red' } );
  const b = new Rectangle( 0, 0, 50, 50, { fill: 'blue' } );
  const c = new DOM( document.createElement( 'div' ) );
  const d = new Rectangle( 100, 0, 100, 50, { fill: 'red' } );
  const e = new Rectangle( 100, 0, 50, 50, { fill: 'blue' } );

  const f = new Rectangle( 0, 50, 100, 50, { fill: 'green' } );
  const g = new DOM( document.createElement( 'div' ) );

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
  display.dispose();

} );

QUnit.test( 'SVG group disposal issue (GitHub Issue #354) A', assert => {
  const scene = new Node();
  const display = new Display( scene );

  const node = new Node( {
    renderer: 'svg',
    cssTransform: true
  } );
  const rect = new Rectangle( 0, 0, 100, 50, { fill: 'red' } );

  scene.addChild( node );
  node.addChild( rect );
  display.updateDisplay();

  scene.removeChild( node );
  display.updateDisplay();

  assert.ok( true, 'so we have at least 1 test in this set' );
  display.dispose();

} );

QUnit.test( 'SVG group disposal issue (GitHub Issue #354) B', assert => {
  const scene = new Node();
  const display = new Display( scene );

  const node = new Node();
  const rect = new Rectangle( 0, 0, 100, 50, {
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
  display.dispose();
} );

QUnit.test( 'Empty path display test', assert => {
  const scene = new Node();
  const display = new Display( scene );

  scene.addChild( new Path( null ) );
  display.updateDisplay();

  assert.ok( true, 'so we have at least 1 test in this set' );
  display.dispose();
} );

QUnit.test( 'Double remove related to #392', assert => {
  const scene = new Node();
  const display = new Display( scene );

  display.updateDisplay();

  const n1 = new Node();
  const n2 = new Node();
  scene.addChild( n1 );
  n1.addChild( n2 );
  scene.addChild( n2 ); // so the tree has a reference to the Node that we can trigger the failure on

  display.updateDisplay();

  scene.removeChild( n1 );
  n1.removeChild( n2 );

  display.updateDisplay();

  assert.ok( true, 'so we have at least 1 test in this set' );
  display.dispose();
} );