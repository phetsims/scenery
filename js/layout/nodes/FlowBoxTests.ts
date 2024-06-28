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

import Rectangle, { RectangleOptions } from '../../nodes/Rectangle.js';
import VBox from './VBox.js';
import HBox from './HBox.js';
import Utils from '../../../../dot/js/Utils.js';

const RECT_WIDTH = 100;
const RECT_HEIGHT = 25;

const aboutEqual = ( a: number, b: number, epsilon = 0.0001 ) => {
  return Utils.equalsEpsilon( a, b, epsilon );
};

const createRectangles = ( count: number, indexToOptions?: ( index: number ) => RectangleOptions ) => {
  return _.times( count, ( index: number ) => {
    const options = indexToOptions ? indexToOptions( index ) : {};
    return new Rectangle( 0, 0, RECT_WIDTH, RECT_HEIGHT, options );
  } );
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
} );

QUnit.test( 'Invisible children', assert => {
  const [ c, d, e ] = createRectangles( 3 );
  d.visible = false;

  // Invisible Nodes should not be included in layout bounds by default.
  const hBoxInvisible = new HBox( { children: [ c, d, e ] } );

  assert.equal( hBoxInvisible.width, RECT_WIDTH * 2, 'width should not include invisible node' );
  assert.equal( c.right, e.left, 'c.right === e.left for middle Node invisible in HBox' );

  // Invisible Nodes can be included in layout bounds if specified.
  hBoxInvisible.setExcludeInvisibleChildrenFromBounds( false );
  assert.equal( hBoxInvisible.width, RECT_WIDTH * 3, 'width should include invisible node' );
  assert.notEqual( c.right, e.left, 'c.right !== e.left when invisible node is included in HBox bounds' );
  assert.equal( c.right, d.left, 'c.right === d.left when invisible node is included in HBox bounds' );
  assert.equal( d.right, e.left, 'd.right === e.left when invisible node is included in HBox bounds' );
  hBoxInvisible.dispose();
} );

QUnit.test( 'Children that grow, stretch, and have size constraints', assert => {
  const [ a, b, c ] = createRectangles( 3, index => {

    // Make these rectangles sizable so that they can grow/stretch for these tests
    return {
      sizable: true,
      localMinimumWidth: RECT_WIDTH,
      localMinimumHeight: RECT_HEIGHT
    };
  } );

  // initial test
  const hBox = new HBox( { children: [ a, b, c ] } );
  assert.equal( hBox.width, RECT_WIDTH * 3, 'width should be sum of children widths' );

  // Make it larger than its contents to test
  hBox.preferredWidth = RECT_WIDTH * 6;
  assert.equal( hBox.width, RECT_WIDTH * 6, 'width should take up preferred width' );
  assert.equal( b.width, RECT_WIDTH, 'b.width should be RECT_WIDTH' );

  // Make b grow to take up all remaining space
  b.layoutOptions = { grow: 1 };
  assert.equal( b.width, RECT_WIDTH * 4, 'b.width should be RECT_WIDTH * 4' );

  // Make a grow and extra space should be shared between a and b
  a.layoutOptions = { grow: 1 };
  assert.equal( a.width, RECT_WIDTH * 2.5, 'a.width should be RECT_WIDTH * 2' );
  assert.equal( b.width, RECT_WIDTH * 2.5, 'b.width should be RECT_WIDTH * 4' );
  assert.equal( c.width, RECT_WIDTH, 'c.width should be RECT_WIDTH' );

  // make c grow and extra space should be shared between all three
  c.layoutOptions = { grow: 1 };
  assert.equal( a.width, RECT_WIDTH * 2, 'a.width should be RECT_WIDTH * 2' );
  assert.equal( b.width, RECT_WIDTH * 2, 'b.width should be RECT_WIDTH * 2' );
  assert.equal( c.width, RECT_WIDTH * 2, 'c.width should be RECT_WIDTH * 2' );

  // Double c's grow value and it should take up proportionally more space - each should tak up the minimum width
  // plus its proportion of teh remaining space as specified by grow.
  c.layoutOptions = { grow: 2 };

  // grow lets nodes take up the remaining EXTRA space, so the distribution should be:
  const extraSpace = RECT_WIDTH * 6 - RECT_WIDTH * 3; // preferred width - sum of minimum widths
  const expectedAWidth = RECT_WIDTH + extraSpace / 4; // distribution of grow values
  const expectedBWidth = RECT_WIDTH + extraSpace / 4;
  const expectedCWidth = RECT_WIDTH + extraSpace / 2;

  assert.equal( a.width, expectedAWidth, 'a.width should be RECT_WIDTH' );
  assert.equal( b.width, expectedBWidth, 'b.width should be RECT_WIDTH' );
  assert.equal( c.width, expectedCWidth, 'c.width should be RECT_WIDTH' );

  //---------------------------------------------------------------------------------
  // stretch
  //---------------------------------------------------------------------------------
  hBox.preferredHeight = RECT_HEIGHT * 3; // extend height of the container
  assert.equal( a.height, RECT_HEIGHT, 'a.height should be RECT_HEIGHT before stretch' );
  a.layoutOptions = { stretch: true };
  assert.equal( a.height, RECT_HEIGHT * 3, 'a.height should be RECT_HEIGHT * 3 after stretch' );

  //---------------------------------------------------------------------------------
  // size constraints
  //---------------------------------------------------------------------------------
  a.layoutOptions = { stretch: false, grow: 1, maxContentWidth: RECT_WIDTH, maxContentHeight: RECT_HEIGHT };
  b.layoutOptions = { stretch: true, grow: 1 };
  c.layoutOptions = { stretch: false, grow: 1 };

  hBox.preferredWidth = RECT_WIDTH * 10;
  hBox.preferredHeight = RECT_HEIGHT * 10;

  const remainingWidth = RECT_WIDTH * 10 - RECT_WIDTH; // the preferred width minus the constrained width of rect a

  assert.equal( a.width, RECT_WIDTH, 'a.width should be RECT_WIDTH because of maxContentWidth, even though it grows' );
  assert.equal( a.height, RECT_HEIGHT, 'a.height should be RECT_HEIGHT because of maxContentHeight, even though it stretches' );

  assert.equal( b.width, remainingWidth / 2, 'b.width should be half of the remaining width' );
  assert.equal( b.height, RECT_HEIGHT * 10, 'b.height should be RECT_HEIGHT * 10 because it stretches' );

  assert.equal( c.width, remainingWidth / 2, 'c.width should be half of the remaining width' );
  assert.equal( c.height, RECT_HEIGHT, 'c.height should be RECT_HEIGHT because it doesnt stretch' );

  //---------------------------------------------------------------------------------
  // size constraints on the container
  //---------------------------------------------------------------------------------
  const [ d, e ] = createRectangles( 3 );
  const [ f, g ] = createRectangles( 2 );

  const hBoxWithConstraint = new HBox( {
    layoutOptions: {
      minContentWidth: RECT_WIDTH * 4
    },
    children: [ d, e ]
  } );

  const vBoxWithConstraint = new VBox( {
    layoutOptions: {
      minContentWidth: RECT_WIDTH * 4
    },
    children: [ f, g ]
  } );

  const combinedBox = new HBox( {
    children: [ hBoxWithConstraint, vBoxWithConstraint ]
  } );

  assert.equal( combinedBox.width, RECT_WIDTH * 8, 'width should be sum of children minContentWidths (applied to all cells)' );
} );

QUnit.test( 'Justify tests', assert => {
  const [ a, b, c, d ] = createRectangles( 4 );

  const hBox = new HBox( { children: [ a, b, c, d ] } );
  assert.equal( hBox.width, 4 * RECT_WIDTH, 'width should be sum of children widths' );

  // Double the preferred width of the container to play with justify effects
  hBox.preferredWidth = RECT_WIDTH * 8;

  assert.equal( hBox.width, RECT_WIDTH * 8, 'width should be the preferred width' );

  //---------------------------------------------------------------------------------
  // justify left
  //---------------------------------------------------------------------------------
  hBox.justify = 'left';
  assert.equal( a.left, hBox.left, 'a.left should be hBox.left' );
  assert.equal( a.right, b.left, 'a.right should be b.left' );
  assert.equal( b.right, c.left, 'b.right should be c.left' );
  assert.equal( c.right, d.left, 'c.right should be d.left' );
  assert.equal( d.right, 4 * RECT_WIDTH, 'd.right should be 4 * RECT_WIDTH' );

  //---------------------------------------------------------------------------------
  // justify right
  //---------------------------------------------------------------------------------
  hBox.justify = 'right';
  assert.equal( a.left, 4 * RECT_WIDTH, 'a.left should be 4 * RECT_WIDTH' );
  assert.equal( b.left, a.right, 'b.left should be a.right' );
  assert.equal( c.left, b.right, 'c.left should be b.right' );
  assert.equal( d.left, c.right, 'd.left should be c.right' );
  assert.equal( d.right, hBox.right, 'd.right should be hBox.right' );

  //---------------------------------------------------------------------------------
  // justify spaceBetween
  //---------------------------------------------------------------------------------
  hBox.justify = 'spaceBetween';
  assert.equal( a.left, hBox.left, 'a.left should be hBox.left' );
  assert.equal( d.right, hBox.right, 'd.right should be hBox.right' );
  assert.ok( aboutEqual( b.left - a.right, c.left - b.right ), 'space between a and b should be equal to space between b and c' );
  assert.ok( aboutEqual( c.left - b.right, d.left - c.right ), 'space between b and c should be equal to space between c and d' );

  //---------------------------------------------------------------------------------
  // justify spaceAround
  //---------------------------------------------------------------------------------
  hBox.justify = 'spaceAround';

  // space around has half the space on the outside of the first and last nodes, and the other half between each pair
  // of nodes
  const totalSpace = hBox.width - 4 * RECT_WIDTH;
  const sideSpacing = totalSpace / 4 / 2; // Each Node gets half space to the left and right

  assert.ok( aboutEqual( a.left, hBox.left + sideSpacing ), 'a.left should be hBox.left + spaceAround' );
  assert.ok( aboutEqual( a.right + sideSpacing * 2, b.left ), 'a.right + sideSpacing * 2 should be b.left' );
  assert.ok( aboutEqual( b.right + sideSpacing * 2, c.left ), 'b.right + sideSpacing * 2 should be c.left' );
  assert.ok( aboutEqual( c.right + sideSpacing * 2, d.left ), 'c.right + sideSpacing * 2 should be d.left' );
  assert.ok( aboutEqual( d.right, hBox.right - sideSpacing ), 'd.right should be hBox.right - spaceAround' );

  //---------------------------------------------------------------------------------
  // justify spaceEvenly
  //---------------------------------------------------------------------------------
  hBox.justify = 'spaceEvenly';

  // space evenly has equal space between each pair of nodes and on the outside of the first and last nodes
  const spaceBetween = totalSpace / 5; // 4 spaces between 5 nodes

  assert.ok( aboutEqual( a.left, hBox.left + spaceBetween ), 'a.left should be hBox.left + spaceEvenly' );
  assert.ok( aboutEqual( a.right + spaceBetween, b.left ), 'a.right + spaceBetween should be b.left' );
  assert.ok( aboutEqual( b.right + spaceBetween, c.left ), 'b.right + spaceBetween should be c.left' );
  assert.ok( aboutEqual( c.right + spaceBetween, d.left ), 'c.right + spaceBetween should be d.left' );
  assert.ok( aboutEqual( d.right, hBox.right - spaceBetween ), 'd.right should be hBox.right - spaceEvenly' );

  //---------------------------------------------------------------------------------
  // justify center
  //---------------------------------------------------------------------------------
  hBox.justify = 'center';

  const remainingSpace = hBox.width - 4 * RECT_WIDTH;
  const halfRemainingSpace = remainingSpace / 2;

  assert.ok( aboutEqual( a.left, hBox.left + halfRemainingSpace ), 'a.left should be hBox.left + halfRemainingSpace' );
  assert.equal( a.right, b.left, 'a.right should be b.left' );
  assert.equal( b.right, c.left, 'b.right should be c.left' );
  assert.equal( c.right, d.left, 'c.right should be d.left' );
  assert.ok( aboutEqual( d.right, hBox.right - halfRemainingSpace ), 'd.right should be hBox.right - halfRemainingSpace' );
} );

QUnit.test( 'Wrap tests', assert => {
  const [ a, b, c, d ] = createRectangles( 4 );
  const hBox = new HBox( { children: [ a, b, c, d ], wrap: true } );

  assert.equal( hBox.width, 4 * RECT_WIDTH, 'width should be sum of children widths' );
  assert.equal( hBox.height, RECT_HEIGHT, 'height should be RECT_HEIGHT' );

  // restrict the preferred width of the container to test wrap
  hBox.preferredWidth = RECT_WIDTH * 2;

  assert.equal( hBox.width, RECT_WIDTH * 2, 'width should be the preferred width' );
  assert.equal( hBox.height, RECT_HEIGHT * 2, 'height should be larger due to wrap' );

  // make the container even smaller to test wrap
  hBox.preferredWidth = RECT_WIDTH;
  assert.equal( hBox.width, RECT_WIDTH, 'width should be the preferred width' );
  assert.equal( hBox.height, RECT_HEIGHT * 4, 'height should be larger due to wrap' );
} );

QUnit.test( 'Align tests', assert => {
  const a = new Rectangle( 0, 0, RECT_WIDTH, 10 );
  const b = new Rectangle( 0, 0, RECT_WIDTH, 20 );
  const c = new Rectangle( 0, 0, RECT_WIDTH, 30 );
  const d = new Rectangle( 0, 0, RECT_WIDTH, 40 );

  const hBox = new HBox( { children: [ a, b, c, d ] } );

  //---------------------------------------------------------------------------------
  // align top
  //---------------------------------------------------------------------------------
  hBox.align = 'top';

  assert.equal( a.top, hBox.top, 'a.top should be hBox.top (align top)' );
  assert.equal( b.top, hBox.top, 'b.top should be hBox.top (align top)' );
  assert.equal( c.top, hBox.top, 'c.top should be hBox.top (align top)' );
  assert.equal( d.top, hBox.top, 'd.top should be hBox.top (align top)' );
  assert.notEqual( a.bottom, b.bottom, 'a.bottom should not be b.bottom (align top)' );
  assert.notEqual( b.bottom, c.bottom, 'b.bottom should not be c.bottom (align top)' );
  assert.notEqual( c.bottom, d.bottom, 'c.bottom should not be d.bottom (align top)' );

  //---------------------------------------------------------------------------------
  // align bottom
  //---------------------------------------------------------------------------------
  hBox.align = 'bottom';

  assert.equal( a.bottom, hBox.bottom, 'a.bottom should be hBox.bottom (align bottom)' );
  assert.equal( b.bottom, hBox.bottom, 'b.bottom should be hBox.bottom (align bottom)' );
  assert.equal( c.bottom, hBox.bottom, 'c.bottom should be hBox.bottom (align bottom)' );
  assert.equal( d.bottom, hBox.bottom, 'd.bottom should be hBox.bottom (align bottom)' );
  assert.notEqual( a.top, b.top, 'a.top should not be b.top (align bottom)' );
  assert.notEqual( b.top, c.top, 'b.top should not be c.top (align bottom)' );
  assert.notEqual( c.top, d.top, 'c.top should not be d.top (align bottom)' );

  //---------------------------------------------------------------------------------
  // align center
  //---------------------------------------------------------------------------------
  hBox.align = 'center';

  assert.equal( a.centerY, hBox.centerY, 'a.centerY should be hBox.centerY (align center)' );
  assert.equal( b.centerY, hBox.centerY, 'b.centerY should be hBox.centerY (align center)' );
  assert.equal( c.centerY, hBox.centerY, 'c.centerY should be hBox.centerY (align center)' );
  assert.equal( d.centerY, hBox.centerY, 'd.centerY should be hBox.centerY (align center)' );

  //---------------------------------------------------------------------------------
  // align origin
  //---------------------------------------------------------------------------------
  hBox.align = 'origin';

  // rectangle origins at top left
  assert.equal( a.top, hBox.top, 'a.top should be hBox.top (align origin)' );
  assert.equal( b.top, hBox.top, 'b.top should be hBox.top (align origin)' );
  assert.equal( c.top, hBox.top, 'c.top should be hBox.top (align origin)' );
  assert.equal( d.top, hBox.top, 'd.top should be hBox.top (align origin)' );
} );