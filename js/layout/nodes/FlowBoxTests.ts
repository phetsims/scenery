// Copyright 2024, University of Colorado Boulder

/**
 * Tests for FlowBox. Covering its various features such as:
 *
 *  - basic layout
 *  - resizing of cells
 *  - grow for cells
 *  - stretch for cells
 *  - constraining cell sizes
 *  - justify
 *  - wrap
 *  - align
 *  - justifyLines
 *  - spacing
 *  - lineSpacing
 *  - margins
 *  - per-cell layout options
 *  - separators
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Rectangle from '../../nodes/Rectangle.js';
import VBox from './VBox.js';
import HBox from './HBox.js';

const RECT_WIDTH = 100;
const RECT_HEIGHT = 25;

const createRectangles = ( count: number ) => {
  return _.times( count, () => new Rectangle( 0, 0, RECT_WIDTH, RECT_HEIGHT, { fill: 'red' } ) );
};

QUnit.module( 'FlowBox' );

QUnit.test( 'Basic HBox/VBox tests', assert => {

  const [ a, b ] = createRectangles( 2 );

  const hBox = new HBox( { children: [ a, b ] } );
  hBox.validateBounds();

  assert.equal( a.right, b.left, 'a.right === b.left for hBox' );
  assert.equal( b.left, a.width, 'b.left === a.width for hBox' );

  // translate box and make sure layout is correct
  hBox.left = 200;
  hBox.validateBounds();
  assert.equal( b.globalBounds.left, 200 + RECT_WIDTH, 'b.globalBounds.left === 200 + RECT_WIDTH' );
  hBox.dispose();

  const vBox = new VBox( { children: [ a, b ] } );
  vBox.validateBounds();

  assert.equal( a.bottom, b.top, 'a.bottom === b.top for vBox' );
  assert.equal( b.top, a.height, 'b.top === a.height for vBox' );

  // translate box and make sure layout is correct
  vBox.top = 200;
  vBox.validateBounds();
  assert.equal( b.globalBounds.top, 200 + RECT_HEIGHT, 'b.globalBounds.top === 200 + RECT_HEIGHT' );
  vBox.dispose();
} );

QUnit.test( 'FlowBox cell resizing', assert => {

  let [ a, b ] = createRectangles( 2 );
  const hBox = new HBox( { children: [ a, b ] } );
  hBox.validateBounds();

  // resize a and make sure layout is correct
  a.rectWidth = RECT_WIDTH * 2;
  hBox.validateBounds();
  assert.equal( a.right, b.left, 'a.right === b.left for hBox after resize' );
  assert.equal( b.left, a.width, 'b.left === a.width for hBox after resize' );
  assert.equal( b.left, RECT_WIDTH * 2, 'b.left === RECT_WIDTH * 2 for hBox after resize' );
  hBox.dispose();

  const vBox = new VBox( { children: [ a, b ] } );
  vBox.validateBounds();

  // resize a and make sure layout is correct
  a.rectHeight = RECT_HEIGHT * 2;
  vBox.validateBounds();
  assert.equal( a.bottom, b.top, 'a.bottom === b.top for vBox after resize' );
  assert.equal( b.top, a.height, 'b.top === a.height for vBox after resize' );
  assert.equal( b.top, RECT_HEIGHT * 2, 'b.top === RECT_WIDTH * 2 for vBox after resize' );
  vBox.dispose();

  //---------------------------------------------------------------------------------
  // Tests that disable resizing
  //---------------------------------------------------------------------------------
  [ a, b ] = createRectangles( 2 );
  const hBoxNoResize = new HBox( { children: [ a, b ], resize: false } );

  // resize a and make sure layout is correct - it should not adjust
  a.rectWidth = RECT_WIDTH * 2;
  hBoxNoResize.validateBounds();
  assert.equal( a.right, RECT_WIDTH * 2, 'a.right === RECT_WIDTH * 2 for hBoxNoResize after resize' );
  assert.equal( b.left, RECT_WIDTH, 'b.left === RECT_WIDTH for hBox after resize' );
  hBoxNoResize.dispose();

  const vBoxNoResize = new VBox( { children: [ a, b ], resize: false } );

  // resize a and make sure layout is correct - it should not adjust
  a.rectHeight = RECT_HEIGHT * 2;
  vBoxNoResize.validateBounds();
  assert.equal( a.bottom, RECT_HEIGHT * 2, 'a.bottom === RECT_HEIGHT * 2 for vBoxNoResize after resize' );
  assert.equal( b.top, RECT_HEIGHT, 'b.top === RECT_HEIGHT for vBox after resize' );
  vBoxNoResize.dispose();

  //---------------------------------------------------------------------------------
  // Tests involving invisible children
  //---------------------------------------------------------------------------------
  // let [ c, d, e ] = createRectangles( 3 );
  // d.visible = false;
  //
  // // Invisible Nodes should not be included in layout bounds by default.
  // const hBoxInvisible = new HBox( { children: [ c, d, e ] } );
} );