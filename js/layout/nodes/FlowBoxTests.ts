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
import LayoutTestUtils from '../LayoutTestUtils.js';
import HBox from './HBox.js';
import VBox from './VBox.js';
import VSeparator from './VSeparator.js';

const RECT_WIDTH = LayoutTestUtils.RECT_WIDTH;
const RECT_HEIGHT = LayoutTestUtils.RECT_HEIGHT;

QUnit.module( 'FlowBox' );

QUnit.test( 'Basic HBox/VBox tests', assert => {

  const [ a, b ] = LayoutTestUtils.createRectangles( 2 );

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

  let [ a, b ] = LayoutTestUtils.createRectangles( 2 );
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
  [ a, b ] = LayoutTestUtils.createRectangles( 2 );
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
  const [ c, d, e ] = LayoutTestUtils.createRectangles( 3 );
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
  const [ a, b, c ] = LayoutTestUtils.createRectangles( 3, index => {

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
  const [ d, e ] = LayoutTestUtils.createRectangles( 3 );
  const [ f, g ] = LayoutTestUtils.createRectangles( 2 );

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
  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );

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
  assert.ok( LayoutTestUtils.aboutEqual( b.left - a.right, c.left - b.right ), 'space between a and b should be equal to space between b and c' );
  assert.ok( LayoutTestUtils.aboutEqual( c.left - b.right, d.left - c.right ), 'space between b and c should be equal to space between c and d' );

  //---------------------------------------------------------------------------------
  // justify spaceAround
  //---------------------------------------------------------------------------------
  hBox.justify = 'spaceAround';

  // space around has half the space on the outside of the first and last nodes, and the other half between each pair
  // of nodes
  const totalSpace = hBox.width - 4 * RECT_WIDTH;
  const sideSpacing = totalSpace / 4 / 2; // Each Node gets half space to the left and right

  assert.ok( LayoutTestUtils.aboutEqual( a.left, hBox.left + sideSpacing ), 'a.left should be hBox.left + spaceAround' );
  assert.ok( LayoutTestUtils.aboutEqual( a.right + sideSpacing * 2, b.left ), 'a.right + sideSpacing * 2 should be b.left' );
  assert.ok( LayoutTestUtils.aboutEqual( b.right + sideSpacing * 2, c.left ), 'b.right + sideSpacing * 2 should be c.left' );
  assert.ok( LayoutTestUtils.aboutEqual( c.right + sideSpacing * 2, d.left ), 'c.right + sideSpacing * 2 should be d.left' );
  assert.ok( LayoutTestUtils.aboutEqual( d.right, hBox.right - sideSpacing ), 'd.right should be hBox.right - spaceAround' );

  //---------------------------------------------------------------------------------
  // justify spaceEvenly
  //---------------------------------------------------------------------------------
  hBox.justify = 'spaceEvenly';

  // space evenly has equal space between each pair of nodes and on the outside of the first and last nodes
  const spaceBetween = totalSpace / 5; // 4 spaces between 5 nodes

  assert.ok( LayoutTestUtils.aboutEqual( a.left, hBox.left + spaceBetween ), 'a.left should be hBox.left + spaceEvenly' );
  assert.ok( LayoutTestUtils.aboutEqual( a.right + spaceBetween, b.left ), 'a.right + spaceBetween should be b.left' );
  assert.ok( LayoutTestUtils.aboutEqual( b.right + spaceBetween, c.left ), 'b.right + spaceBetween should be c.left' );
  assert.ok( LayoutTestUtils.aboutEqual( c.right + spaceBetween, d.left ), 'c.right + spaceBetween should be d.left' );
  assert.ok( LayoutTestUtils.aboutEqual( d.right, hBox.right - spaceBetween ), 'd.right should be hBox.right - spaceEvenly' );

  //---------------------------------------------------------------------------------
  // justify center
  //---------------------------------------------------------------------------------
  hBox.justify = 'center';

  const remainingSpace = hBox.width - 4 * RECT_WIDTH;
  const halfRemainingSpace = remainingSpace / 2;

  assert.ok( LayoutTestUtils.aboutEqual( a.left, hBox.left + halfRemainingSpace ), 'a.left should be hBox.left + halfRemainingSpace' );
  assert.equal( a.right, b.left, 'a.right should be b.left' );
  assert.equal( b.right, c.left, 'b.right should be c.left' );
  assert.equal( c.right, d.left, 'c.right should be d.left' );
  assert.ok( LayoutTestUtils.aboutEqual( d.right, hBox.right - halfRemainingSpace ), 'd.right should be hBox.right - halfRemainingSpace' );
} );

QUnit.test( 'Wrap tests', assert => {
  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );
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


QUnit.test( 'Justify Lines Tests', assert => {

  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );
  const hBox = new HBox( {
    children: [ a, b, c, d ],

    // so that rectangles will stack on the secondary axis
    preferredWidth: RECT_WIDTH,
    wrap: true,

    // so there is plenty of room on the secondary axis to test justifyLines
    preferredHeight: RECT_HEIGHT * 8
  } );

  //---------------------------------------------------------------------------------
  // justifyLines top
  //---------------------------------------------------------------------------------
  hBox.justifyLines = 'top';

  assert.equal( a.top, hBox.top, 'a.top should be hBox.top (justifyLines top)' );
  assert.equal( b.top, a.bottom, 'b.top should be a.bottom (justifyLines top)' );
  assert.equal( c.top, b.bottom, 'c.top should be b.bottom (justifyLines top)' );
  assert.equal( d.top, c.bottom, 'd.top should be c.bottom (justifyLines top)' );

  //---------------------------------------------------------------------------------
  // justifyLines bottom
  //---------------------------------------------------------------------------------
  hBox.justifyLines = 'bottom';

  assert.equal( d.bottom, hBox.bottom, 'd.bottom should be hBox.bottom (justifyLines bottom)' );
  assert.equal( c.bottom, d.top, 'c.bottom should be d.top (justifyLines bottom)' );
  assert.equal( b.bottom, c.top, 'b.bottom should be c.top (justifyLines bottom)' );
  assert.equal( a.bottom, b.top, 'a.bottom should be b.top (justifyLines bottom)' );

  //---------------------------------------------------------------------------------
  // justifyLines center
  //---------------------------------------------------------------------------------
  hBox.justifyLines = 'center';

  assert.equal( a.top, hBox.height / 2 - RECT_HEIGHT * 2, 'a.top should be half the height minus half the height of the rectangles' );
  assert.equal( b.top, a.bottom, 'b.top should be a.bottom (justifyLines center)' );
  assert.equal( c.top, b.bottom, 'c.top should be b.bottom (justifyLines center)' );
  assert.equal( d.top, c.bottom, 'd.top should be c.bottom (justifyLines center)' );

  //---------------------------------------------------------------------------------
  // justifyLines spaceBetween
  //---------------------------------------------------------------------------------
  hBox.justifyLines = 'spaceBetween';

  assert.ok( LayoutTestUtils.aboutEqual( a.top, hBox.top ), 'a.top should be hBox.top (justifyLines spaceBetween)' );
  assert.ok( LayoutTestUtils.aboutEqual( d.bottom, hBox.bottom ), 'd.bottom should be hBox.bottom (justifyLines spaceBetween)' );
  assert.ok( LayoutTestUtils.aboutEqual( b.top - a.bottom, c.top - b.bottom ), 'space between a and b should be equal to space between b and c' );
  assert.ok( LayoutTestUtils.aboutEqual( c.top - b.bottom, d.top - c.bottom ), 'space between b and c should be equal to space between c and d' );

  //---------------------------------------------------------------------------------
  // justifyLines spaceAround
  //---------------------------------------------------------------------------------
  hBox.justifyLines = 'spaceAround';

  const totalSpace = hBox.height - 4 * RECT_HEIGHT;
  const sideSpacing = totalSpace / 4 / 2; // Each Node gets half space to the top and bottom

  assert.ok( LayoutTestUtils.aboutEqual( a.top, hBox.top + sideSpacing ), 'a.top should be hBox.top + spaceAround' );
  assert.ok( LayoutTestUtils.aboutEqual( a.bottom + sideSpacing * 2, b.top ), 'a.bottom + sideSpacing * 2 should be b.top' );
  assert.ok( LayoutTestUtils.aboutEqual( b.bottom + sideSpacing * 2, c.top ), 'b.bottom + sideSpacing * 2 should be c.top' );
  assert.ok( LayoutTestUtils.aboutEqual( c.bottom + sideSpacing * 2, d.top ), 'c.bottom + sideSpacing * 2 should be d.top' );
  assert.ok( LayoutTestUtils.aboutEqual( d.bottom, hBox.bottom - sideSpacing ), 'd.bottom should be hBox.bottom - spaceAround' );

  //---------------------------------------------------------------------------------
  // justifyLines spaceEvenly
  //---------------------------------------------------------------------------------
  hBox.justifyLines = 'spaceEvenly';

  const spaceBetween = totalSpace / 5; // 4 spaces between 5 nodes

  assert.ok( LayoutTestUtils.aboutEqual( a.top, hBox.top + spaceBetween ), 'a.top should be hBox.top + spaceEvenly' );
  assert.ok( LayoutTestUtils.aboutEqual( a.bottom + spaceBetween, b.top ), 'a.bottom + spaceBetween should be b.top' );
  assert.ok( LayoutTestUtils.aboutEqual( b.bottom + spaceBetween, c.top ), 'b.bottom + spaceBetween should be c.top' );
  assert.ok( LayoutTestUtils.aboutEqual( c.bottom + spaceBetween, d.top ), 'c.bottom + spaceBetween should be d.top' );
  assert.ok( LayoutTestUtils.aboutEqual( d.bottom, hBox.bottom - spaceBetween ), 'd.bottom should be hBox.bottom - spaceEvenly' );


  //---------------------------------------------------------------------------------
  // justifyLines null (default to stretch (same as spaceAround))
  //---------------------------------------------------------------------------------
  hBox.justifyLines = null;

  assert.ok( LayoutTestUtils.aboutEqual( a.top, hBox.top + sideSpacing ), 'a.top should be hBox.top + spaceAround' );
  assert.ok( LayoutTestUtils.aboutEqual( a.bottom + sideSpacing * 2, b.top ), 'a.bottom + sideSpacing * 2 should be b.top' );
  assert.ok( LayoutTestUtils.aboutEqual( b.bottom + sideSpacing * 2, c.top ), 'b.bottom + sideSpacing * 2 should be c.top' );
  assert.ok( LayoutTestUtils.aboutEqual( c.bottom + sideSpacing * 2, d.top ), 'c.bottom + sideSpacing * 2 should be d.top' );
  assert.ok( LayoutTestUtils.aboutEqual( d.bottom, hBox.bottom - sideSpacing ), 'd.bottom should be hBox.bottom - spaceAround' );
} );

QUnit.test( 'Spacing tests', assert => {
  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );

  const hBox = new HBox( { children: [ a, b, c, d ], spacing: 10 } );

  assert.ok( LayoutTestUtils.aboutEqual( b.left - a.right, 10 ), 'b.left - a.right should be 10' );
  assert.ok( LayoutTestUtils.aboutEqual( c.left - b.right, 10 ), 'c.left - b.right should be 10' );
  assert.ok( LayoutTestUtils.aboutEqual( d.left - c.right, 10 ), 'd.left - c.right should be 10' );
  assert.ok( LayoutTestUtils.aboutEqual( hBox.width, 4 * RECT_WIDTH + 3 * 10 ), 'width should be sum of children widths plus spacing' );
} );

QUnit.test( 'lineSpacing tests', assert => {

  // Line spacing is the spacing between lines of nodes in a wrap layout
  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );

  const hBox = new HBox( {
    children: [ a, b, c, d ],
    lineSpacing: 10,

    // so that the contents wrap and we can test lineSpacing
    wrap: true,
    preferredWidth: RECT_WIDTH
  } );

  assert.ok( LayoutTestUtils.aboutEqual( a.top, hBox.top ), 'a.top should be hBox.top' );
  assert.ok( LayoutTestUtils.aboutEqual( b.top - a.bottom, 10 ), 'b.top - a.bottom should be 10' );
  assert.ok( LayoutTestUtils.aboutEqual( c.top - b.bottom, 10 ), 'c.top - b.bottom should be 10' );
  assert.ok( LayoutTestUtils.aboutEqual( d.top - c.bottom, 10 ), 'd.top - c.bottom should be 10' );
  assert.ok( LayoutTestUtils.aboutEqual( hBox.height, 4 * RECT_HEIGHT + 3 * 10 ), 'height should be sum of children heights plus lineSpacing' );
} );

QUnit.test( 'Margins tests', assert => {

  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );

  const hBox = new HBox( {
    children: [ a, b, c, d ]
  } );

  //---------------------------------------------------------------------------------
  // margin tests
  //---------------------------------------------------------------------------------
  hBox.margin = 10;

  assert.ok( LayoutTestUtils.aboutEqual( a.left, hBox.left + 10 ), 'a.left should be hBox.left + 10' );
  assert.ok( LayoutTestUtils.aboutEqual( b.left, a.right + 20 ), 'b.left should be a.right + 10' );
  assert.ok( LayoutTestUtils.aboutEqual( c.left, b.right + 20 ), 'c.left should be b.right + 10' );
  assert.ok( LayoutTestUtils.aboutEqual( d.left, c.right + 20 ), 'd.left should be c.right + 10' );
  assert.ok( LayoutTestUtils.aboutEqual( d.right, hBox.right - 10 ), 'd.right should be hBox.right - 10' );
  assert.ok( LayoutTestUtils.aboutEqual( a.top, hBox.top + 10 ), 'a.top should be hBox.top + 10' );
  assert.ok( LayoutTestUtils.aboutEqual( b.top, hBox.top + 10 ), 'b.top should be hBox.top + 10' );
  assert.ok( LayoutTestUtils.aboutEqual( c.top, hBox.top + 10 ), 'c.top should be hBox.top + 10' );
  assert.ok( LayoutTestUtils.aboutEqual( d.top, hBox.top + 10 ), 'd.top should be hBox.top + 10' );

  //---------------------------------------------------------------------------------
  // left margin tests
  //---------------------------------------------------------------------------------
  hBox.margin = 0;
  hBox.leftMargin = 10;

  assert.ok( LayoutTestUtils.aboutEqual( a.left, hBox.left + 10 ), 'a.left should be hBox.left' );
  assert.ok( LayoutTestUtils.aboutEqual( b.left, a.right + 10 ), 'b.left should be a.right + 10' );
  assert.ok( LayoutTestUtils.aboutEqual( c.left, b.right + 10 ), 'c.left should be b.right + 10' );
  assert.ok( LayoutTestUtils.aboutEqual( d.left, c.right + 10 ), 'd.left should be c.right + 10' );
  assert.ok( LayoutTestUtils.aboutEqual( d.right, hBox.right ), 'd.right should be hBox.right' );
} );

QUnit.test( 'Per-cell layout options', assert => {
  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );

  const hBox = new HBox( {
    children: [ a, b, c, d ]
  } );

  //---------------------------------------------------------------------------------
  // per-cel margins
  //---------------------------------------------------------------------------------

  const margin = 10;
  a.layoutOptions = { topMargin: margin };
  d.layoutOptions = { leftMargin: margin };

  assert.ok( LayoutTestUtils.aboutEqual( a.top, hBox.top + margin ), 'a.top should be hBox.top + margin' );
  assert.ok( LayoutTestUtils.aboutEqual( b.top, hBox.top + margin / 2 ), 'hBox dimensions grow but b remains centered by default' );
  assert.ok( LayoutTestUtils.aboutEqual( c.top, hBox.top + margin / 2 ), 'hBox dimensions grow but c remains centered by default' );
  assert.ok( LayoutTestUtils.aboutEqual( d.left, c.right + margin ), 'd.left should be c.left + margin' );
  assert.ok( LayoutTestUtils.aboutEqual( d.right, hBox.right ), 'd.right should be hBox.right' );

  //---------------------------------------------------------------------------------
  // per-cel alignment
  //---------------------------------------------------------------------------------
  hBox.preferredHeight = RECT_HEIGHT * 2; // extend height of the container
  a.layoutOptions = { align: 'top' };
  b.layoutOptions = { align: 'bottom' };
  c.layoutOptions = { align: 'center' };
  d.layoutOptions = {};

  assert.ok( LayoutTestUtils.aboutEqual( a.top, hBox.top ), 'a.top should be hBox.top' );
  assert.ok( LayoutTestUtils.aboutEqual( b.bottom, hBox.bottom ), 'b.bottom should be hBox.bottom' );
  assert.ok( LayoutTestUtils.aboutEqual( c.centerY, hBox.centerY ), 'c.centerY should be hBox.centerY' );
  assert.ok( LayoutTestUtils.aboutEqual( d.centerY, hBox.centerY ), 'd.centerY should be hBox.centerY' );

  //---------------------------------------------------------------------------------
  // cells override the container
  //---------------------------------------------------------------------------------
  hBox.align = 'top';

  a.layoutOptions = { align: 'bottom' };
  b.layoutOptions = { align: 'center' };
  c.layoutOptions = { align: 'bottom' };
  d.layoutOptions = {};

  assert.ok( LayoutTestUtils.aboutEqual( a.bottom, hBox.bottom ), 'a.bottom should be hBox.bottom' );
  assert.ok( LayoutTestUtils.aboutEqual( b.centerY, hBox.centerY ), 'b.centerY should be hBox.centerY' );
  assert.ok( LayoutTestUtils.aboutEqual( c.bottom, hBox.bottom ), 'c.bottom should be hBox.bottom' );
  assert.ok( LayoutTestUtils.aboutEqual( d.top, hBox.top ), 'd.top should be hBox.top' );
} );

QUnit.test( 'Separators', assert => {
  const margin = 5;

  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );
  const hBox = new HBox( {
    margin: margin
  } );

  const verifySeparatorLayout = ( separatorWidth: number ) => {
    assert.ok( LayoutTestUtils.aboutEqual( b.left, a.right + separatorWidth + margin * 4 ), 'b.left should be a.right + separatorWidth + margin * 4 (each side of rectangles + each side of separator)' );
    assert.ok( LayoutTestUtils.aboutEqual( c.left, b.right + separatorWidth + margin * 4 ), 'c.left should be b.right + separatorWidth + margin * 4' );
    assert.ok( LayoutTestUtils.aboutEqual( d.left, c.right + separatorWidth + margin * 4 ), 'd.left should be c.right + separatorWidth + margin * 4' );
  };

  const testSeparator = new VSeparator();

  //---------------------------------------------------------------------------------
  // basic tests
  //---------------------------------------------------------------------------------
  hBox.children = [ a, new VSeparator(), b, new VSeparator(), c, new VSeparator(), d ];
  verifySeparatorLayout( testSeparator.width );

  //---------------------------------------------------------------------------------
  // duplicate separators are removed
  //---------------------------------------------------------------------------------
  hBox.children = [ a, new VSeparator(), new VSeparator(), b, new VSeparator(), c, new VSeparator(), new VSeparator(), d ];
  verifySeparatorLayout( testSeparator.width );

  //---------------------------------------------------------------------------------
  // separators at the ends are removed
  //---------------------------------------------------------------------------------
  hBox.children = [ new VSeparator(), a, new VSeparator(), b, new VSeparator(), c, new VSeparator(), d, new VSeparator() ];
  verifySeparatorLayout( testSeparator.width );
  assert.ok( LayoutTestUtils.aboutEqual( a.left, hBox.left + margin ), 'a.left should be hBox.left + margin (separator removed)' );
  assert.ok( LayoutTestUtils.aboutEqual( d.right, hBox.right - margin ), 'd.right should be hBox.right - margin (separator removed)' );

  //---------------------------------------------------------------------------------
  // custom separators
  //---------------------------------------------------------------------------------
  const createCustomSeparator = () => new Rectangle( 0, 0, 10, 10, { layoutOptions: { isSeparator: true } } );
  const testCustomSeparator = createCustomSeparator();

  // basic
  hBox.children = [ a, createCustomSeparator(), b, createCustomSeparator(), c, createCustomSeparator(), d ];
  verifySeparatorLayout( testCustomSeparator.width );

  // duplicates removed
  hBox.children = [ a, createCustomSeparator(), createCustomSeparator(), b, createCustomSeparator(), c, createCustomSeparator(), createCustomSeparator(), d ];
  verifySeparatorLayout( testCustomSeparator.width );

  // separators at the ends are removed
  hBox.children = [ createCustomSeparator(), a, createCustomSeparator(), b, createCustomSeparator(), c, createCustomSeparator(), d, createCustomSeparator() ];
  verifySeparatorLayout( testCustomSeparator.width );
  assert.ok( LayoutTestUtils.aboutEqual( a.left, hBox.left + margin ), 'a.left should be hBox.left + margin (separator removed)' );
  assert.ok( LayoutTestUtils.aboutEqual( d.right, hBox.right - margin ), 'd.right should be hBox.right - margin (separator removed)' );
} );