// Copyright 2024, University of Colorado Boulder

import LayoutTestUtils from '../LayoutTestUtils.js';
import GridBox from './GridBox.js';
import Rectangle from '../../nodes/Rectangle.js';

/**
 * Tests for GridBox. Covering its various features such as:
 *
 *  - cell coordinates
 *  - specifying rows, columns, and changing them
 *  - line/cell getters
 *  - grow (auto expand)
 *  - stretch for cells
 *  - sizable
 *  - cell alignment
 *  - horizontal/vertical span
 *  - spacing
 *  - margins
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

const RECT_WIDTH = LayoutTestUtils.RECT_WIDTH;
const RECT_HEIGHT = LayoutTestUtils.RECT_HEIGHT;

QUnit.module( 'GridBox' );

QUnit.test( 'Cell coordinates tests', assert => {

  //---------------------------------------------------------------------------------
  // Check that coordinates are correct from column/row coordinates
  //---------------------------------------------------------------------------------

  // Create a grid with 5 rectangles where each rectangle has a specific cell coordinate
  const [ a, b, c, d, e ] = LayoutTestUtils.createRectangles( 5, index => {

    // Layout options will produce a grid with 3 columns and 5 rows - each rectangle will be in its own row,
    // cycling through the columns
    return {
      layoutOptions: { column: index % 3, row: index }
    };
  } );

  const grid = new GridBox( {
    children: [ a, b, c, d, e ]
  } );

  assert.ok( a.leftTop.equals( grid.leftTop ), 'Rectangle a at cell coordinates (0,0)' );
  assert.ok( b.leftTop.equals( grid.leftTop.plusXY( RECT_WIDTH, RECT_HEIGHT ) ), 'Rectangle b at cell coordinates (1,1)' );
  assert.ok( c.leftTop.equals( grid.leftTop.plusXY( 2 * RECT_WIDTH, 2 * RECT_HEIGHT ) ), 'Rectangle c at cell coordinates (2,2)' );
  assert.ok( d.leftTop.equals( grid.leftTop.plusXY( 0, 3 * RECT_HEIGHT ) ), 'Rectangle d at cell coordinates (0,3)' );
  assert.ok( e.leftTop.equals( grid.leftTop.plusXY( RECT_WIDTH, 4 * RECT_HEIGHT ) ), 'Rectangle e at cell coordinates (1,4)' );

  //---------------------------------------------------------------------------------
  // Skipped rows/columns are collapsed and don't apply
  //---------------------------------------------------------------------------------
  const [ f, g, h ] = LayoutTestUtils.createRectangles( 5, index => {

    // Very large values for column/row, but it should produce a 3x3 grid with rectangles along the diagonal
    return {
      layoutOptions: { column: index * 10, row: index * 10 }
    };
  } );

  const grid2 = new GridBox( {
    children: [ f, g, h ]
  } );

  assert.ok( f.leftTop.equals( grid2.leftTop ), 'Rectangle f at cell coordinates (0,0) ' );
  assert.ok( g.leftTop.equals( grid2.leftTop.plusXY( RECT_WIDTH, RECT_HEIGHT ) ), 'Rectangle g at cell coordinates (1,1) (skipping does not apply)' );
  assert.ok( h.leftTop.equals( grid2.leftTop.plusXY( 2 * RECT_WIDTH, 2 * RECT_HEIGHT ) ), 'Rectangle h at cell coordinates (2,2) (skipping does not apply)' );
} );

QUnit.test( 'Rows and Columns', assert => {

  //---------------------------------------------------------------------------------
  // Construct with rows
  //---------------------------------------------------------------------------------
  const [ a, b, c ] = LayoutTestUtils.createRectangles( 3 );

  const grid1 = new GridBox( {
    rows: [
      [ a, b ],
      [ c ]
    ]
  } );

  assert.ok( a.leftTop.equals( grid1.leftTop ), 'Rectangle a at cell coordinates (0,0) (setting rows)' );
  assert.ok( b.leftTop.equals( grid1.leftTop.plusXY( RECT_WIDTH, 0 ) ), 'Rectangle b at cell coordinates (1,0) (setting rows)' );
  assert.ok( c.leftTop.equals( grid1.leftTop.plusXY( 0, RECT_HEIGHT ) ), 'Rectangle c at cell coordinates (0,1) (setting rows)' );
  grid1.dispose();

  //---------------------------------------------------------------------------------
  // Construct with columns
  //---------------------------------------------------------------------------------
  const [ d, e, f ] = LayoutTestUtils.createRectangles( 3 );

  const grid2 = new GridBox( {
    columns: [
      [ d, e ],
      [ f ]
    ]
  } );

  assert.ok( d.leftTop.equals( grid2.leftTop ), 'Rectangle d at cell coordinates (0,0) (setting columns)' );
  assert.ok( e.leftTop.equals( grid2.leftTop.plusXY( 0, RECT_HEIGHT ) ), 'Rectangle e at cell coordinates (0,1) (setting columns)' );
  assert.ok( f.leftTop.equals( grid2.leftTop.plusXY( RECT_WIDTH, 0 ) ), 'Rectangle f at cell coordinates (1,0) (setting columns)' );
  grid2.dispose();

  //---------------------------------------------------------------------------------
  // autoRows/autoColumns
  //---------------------------------------------------------------------------------
  const grid3 = new GridBox( {
    autoColumns: 2,
    children: [ a, b, c, d, e, f ]
  } );

  assert.ok( a.leftTop.equals( grid3.leftTop ), 'Rectangle a at cell coordinates (0,0) (autoColumns)' );
  assert.ok( b.leftTop.equals( grid3.leftTop.plusXY( RECT_WIDTH, 0 ) ), 'Rectangle b at cell coordinates (1,0) (autoColumns)' );
  assert.ok( c.leftTop.equals( grid3.leftTop.plusXY( 0, RECT_HEIGHT ) ), 'Rectangle c at cell coordinates (0,1) (autoColumns)' );
  assert.ok( d.leftTop.equals( grid3.leftTop.plusXY( RECT_WIDTH, RECT_HEIGHT ) ), 'Rectangle d at cell coordinates (1,1) (autoColumns)' );
  assert.ok( e.leftTop.equals( grid3.leftTop.plusXY( 0, 2 * RECT_HEIGHT ) ), 'Rectangle e at cell coordinates (0,2) (autoColumns)' );
  assert.ok( f.leftTop.equals( grid3.leftTop.plusXY( RECT_WIDTH, 2 * RECT_HEIGHT ) ), 'Rectangle f at cell coordinates (1,2) (autoColumns)' );
  grid3.dispose();

  const grid4 = new GridBox( {
    autoRows: 2,
    children: [ a, b, c, d, e, f ]
  } );

  assert.ok( a.leftTop.equals( grid4.leftTop ), 'Rectangle a at cell coordinates (0,0) (autoRows)' );
  assert.ok( b.leftTop.equals( grid4.leftTop.plusXY( 0, RECT_HEIGHT ) ), 'Rectangle b at cell coordinates (0,1) (autoRows)' );
  assert.ok( c.leftTop.equals( grid4.leftTop.plusXY( RECT_WIDTH, 0 ) ), 'Rectangle c at cell coordinates (1,0) (autoRows)' );
  assert.ok( d.leftTop.equals( grid4.leftTop.plusXY( RECT_WIDTH, RECT_HEIGHT ) ), 'Rectangle d at cell coordinates (1,1) (autoRows)' );
  assert.ok( e.leftTop.equals( grid4.leftTop.plusXY( 2 * RECT_WIDTH, 0 ) ), 'Rectangle e at cell coordinates (2,0) (autoRows)' );
  assert.ok( f.leftTop.equals( grid4.leftTop.plusXY( 2 * RECT_WIDTH, RECT_HEIGHT ) ), 'Rectangle f at cell coordinates (2,1) (autoRows)' );
  grid4.dispose();

  //---------------------------------------------------------------------------------
  // addRow/addColumn
  //---------------------------------------------------------------------------------
  const grid5 = new GridBox();
  grid5.addRow( [ a, b ] );
  grid5.addColumn( [ c, d, e ] );
  grid5.addRow( [ f ] );

  assert.ok( a.leftTop.equals( grid5.leftTop ), 'Rectangle a at cell coordinates (0,0) (addRow)' );
  assert.ok( b.leftTop.equals( grid5.leftTop.plusXY( RECT_WIDTH, 0 ) ), 'Rectangle b at cell coordinates (1,0) (addRow)' );

  // columns are added to right of current content
  assert.ok( c.leftTop.equals( grid5.leftTop.plusXY( RECT_WIDTH * 2, 0 ) ), 'Rectangle c at cell coordinates (2,0) (addColumn)' );
  assert.ok( d.leftTop.equals( grid5.leftTop.plusXY( RECT_WIDTH * 2, RECT_HEIGHT ) ), 'Rectangle d at cell coordinates (2,1) (addColumn)' );
  assert.ok( e.leftTop.equals( grid5.leftTop.plusXY( RECT_WIDTH * 2, RECT_HEIGHT * 2 ) ), 'Rectangle e at cell coordinates (2,2) (addColumn)' );

  // rows are added below current content
  assert.ok( f.leftTop.equals( grid5.leftTop.plusXY( 0, RECT_HEIGHT * 3 ) ), 'Rectangle f at cell coordinates (0,2) (addRow)' );
  grid5.dispose();

  //---------------------------------------------------------------------------------
  // insertRow/insertColumn
  //---------------------------------------------------------------------------------
  const grid6 = new GridBox();

  grid6.insertRow( 0, [ a, b ] );
  grid6.insertColumn( 1, [ c, d, e ] );

  assert.ok( a.leftTop.equals( grid6.leftTop ), 'Rectangle a at cell coordinates (0,0) (insertRow)' );
  assert.ok( b.leftTop.equals( grid6.leftTop.plusXY( RECT_WIDTH * 2, 0 ) ), 'Rectangle b at cell coordinates (2,0) (insertColumn/insertColumn)' );
  assert.ok( c.leftTop.equals( grid6.leftTop.plusXY( RECT_WIDTH, 0 ) ), 'Rectangle c at cell coordinates (1,0) (insertColumn)' );
  assert.ok( d.leftTop.equals( grid6.leftTop.plusXY( RECT_WIDTH, RECT_HEIGHT ) ), 'Rectangle d at cell coordinates (1,1) (insertColumn)' );
  assert.ok( e.leftTop.equals( grid6.leftTop.plusXY( RECT_WIDTH, RECT_HEIGHT * 2 ) ), 'Rectangle e at cell coordinates (1,2) (insertColumn)' );
  grid6.dispose();

  //---------------------------------------------------------------------------------
  // removeRow/removeColumn
  //---------------------------------------------------------------------------------
  const grid7 = new GridBox( {
    rows: [
      [ a, b ],
      [ c, d, e ],
      [ f ]
    ]
  } );

  grid7.removeRow( 1 );
  grid7.removeColumn( 0 );

  // only b should remain
  assert.ok( b.leftTop.equals( grid7.leftTop ), 'Rectangle b at cell coordinates (0,0) (removeRow/removeColumn)' );
  grid7.dispose();
} );

QUnit.test( 'Line/Cell getters', assert => {
  const [ a, b, c, d, e, f ] = LayoutTestUtils.createRectangles( 6 );
  const grid8 = new GridBox( {
    rows: [
      [ a, b ],
      [ c, d, e ],
      [ f ]
    ]
  } );

  assert.equal( grid8.getRowOfNode( d ), 1, 'getRowOfNode' );
  assert.equal( grid8.getColumnOfNode( d ), 1, 'getColumnOfNode' );

  assert.ok( _.isEqual( grid8.getNodesInRow( grid8.getRowOfNode( d ) ), [ c, d, e ] ), 'getNodesInRow' );
  assert.ok( _.isEqual( grid8.getNodesInColumn( grid8.getColumnOfNode( d ) ), [ b, d ] ), 'getNodesInColumn' );
} );

QUnit.test( 'Grow', assert => {

  //---------------------------------------------------------------------------------
  // GridBox grow
  //---------------------------------------------------------------------------------

  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );

  // much larger than the 4 rectangles to play with grow functionality
  const fullWidth = RECT_WIDTH * 8;
  const fullHeight = RECT_HEIGHT * 8;

  const grid = new GridBox( {
    rows: [ [ a, b ], [ c, d ] ],
    preferredWidth: fullWidth,
    preferredHeight: fullHeight,
    grow: 1
  } );

  assert.ok( grid.width === fullWidth, 'Grid width should be full width' );
  assert.ok( grid.height === fullHeight, 'Grid height should be full height' );

  // similar style to flowbox - extra space should be distributed evenly
  assert.ok( a.centerX = fullWidth / 4, 'Rectangle a should be centered in the x direction' );
  assert.ok( a.centerY = fullHeight / 4, 'Rectangle a should be centered in the y direction' );
  assert.ok( b.centerX = 3 * fullWidth / 4, 'Rectangle b should be centered in the x direction' );
  assert.ok( b.centerY = fullHeight / 4, 'Rectangle b should be centered in the y direction' );
  assert.ok( c.centerX = fullWidth / 4, 'Rectangle c should be centered in the x direction' );
  assert.ok( c.centerY = 3 * fullHeight / 4, 'Rectangle c should be centered in the y direction' );
  assert.ok( d.centerX = 3 * fullWidth / 4, 'Rectangle d should be centered in the x direction' );
  assert.ok( d.centerY = 3 * fullHeight / 4, 'Rectangle d should be centered in the y direction' );

  //---------------------------------------------------------------------------------
  // Cell grow
  //---------------------------------------------------------------------------------
  const [ e, f, g, h ] = LayoutTestUtils.createRectangles( 4 );
  e.layoutOptions = { xGrow: 1 };

  const grid2 = new GridBox( {
    preferredWidth: fullWidth,
    preferredHeight: fullHeight,
    rows: [ [ e, f ], [ g, h ] ]
  } );

  const remainingWidth = fullWidth - RECT_WIDTH * 2;

  assert.ok( e.left === remainingWidth / 2, 'cell for e will grow to take up all extra space' );
  assert.ok( g.left === remainingWidth / 2, 'cell for g will grow to match column for e' );

  assert.ok( f.right === grid2.right, 'all available space given to grow cells' );
  assert.ok( h.right === grid2.right, 'all available space given to grow cells' );
} );

QUnit.test( 'Stretch', assert => {

  const [ a, c, d ] = LayoutTestUtils.createRectangles( 4 );
  const b = new Rectangle( {
    sizable: true,
    localMinimumWidth: RECT_WIDTH,
    localMinimumHeight: RECT_HEIGHT,
    layoutOptions: { grow: 1, stretch: true }
  } );

  // preferred dimensions are very large to test stretching behavior
  const totalWidth = RECT_WIDTH * 8;
  const totalHeight = RECT_HEIGHT * 8;

  // Should look like
  // | [a] | [    b    ]  |
  // | [c] |     [d]      |
  const grid = new GridBox( {
    preferredWidth: totalWidth,
    preferredHeight: totalHeight,
    rows: [ [ a, b ], [ c, d ] ]
  } );

  assert.equal( b.width, grid.width - RECT_WIDTH, 'b should stretch to fill the remaining width' );
  assert.equal( a.left, 0, 'a should be at the left edge' );
  assert.equal( b.left, RECT_WIDTH, 'b should be to the right of a' );
  assert.equal( c.left, 0, 'c should be at the left edge' );
  assert.equal( d.centerX, b.centerX, 'd should be centered with b (column constraint)' );
} );

QUnit.test( 'Sizable', assert => {

  const [ a, b, c, d, f ] = LayoutTestUtils.createRectangles( 6 );
  const e = new Rectangle( {
    heightSizable: true,
    layoutOptions: { grow: 1, stretch: true },
    localMinimumHeight: 50,
    widthSizable: false,
    rectWidth: RECT_WIDTH
  } );

  // extra width/height to make space for sizable rectangle
  const totalWidth = RECT_WIDTH * 8;
  const totalHeight = RECT_HEIGHT * 8;

  const grid = new GridBox( {
    preferredWidth: totalWidth,
    preferredHeight: totalHeight,
    rows: [
      [ a, b, c ],
      [ d, e, f ]
    ]
  } );

  assert.ok( a.left === 0, 'a should be at the left edge' );
  assert.ok( b.centerX === totalWidth / 2, 'b should be centered in the grid' );
  assert.ok( c.right === totalWidth, 'c should be at the right edge' );
  assert.ok( d.left === 0, 'd should be at the left edge' );
  assert.ok( e.centerX === totalWidth / 2, 'e should be centered in the grid' );
  assert.ok( e.width === RECT_WIDTH, 'e should not be sizable in the width' );
  assert.ok( e.height === totalHeight - RECT_HEIGHT, 'e should be sizable in the height' );
  assert.ok( f.right === totalWidth, 'f should be at the right edge' );
  assert.ok( grid.height === totalHeight, 'grid should be full height' );
  assert.ok( grid.width === totalWidth, 'grid should be full width' );
} );

QUnit.test( 'Cell alignment', assert => {

  const [ a, b, c, d, e, f ] = LayoutTestUtils.createRectangles( 6 );
  a.layoutOptions = { xAlign: 'left', yAlign: 'top' };
  c.layoutOptions = { xAlign: 'right', yAlign: 'bottom' };
  e.layoutOptions = { yAlign: 'top' };

  // Additional width to make space for alignment tests with grow
  const fullWidth = RECT_WIDTH * 8;
  const fullHeight = RECT_HEIGHT * 8;

  // Should look like
  // |[a]  |     |     |
  // |     | [b] |     |
  // |     |     |  [c]|
  // |     | [e] |     |
  // | [d] |     | [f] |
  // |     |     |     |
  const grid = new GridBox( {
    grow: 1,
    preferredWidth: fullWidth,
    preferredHeight: fullHeight,
    rows: [ [ a, b, c ], [ d, e, f ] ]
  } );

  assert.ok( grid.width === fullWidth, 'grid should be full width' );
  assert.ok( grid.height === fullHeight, 'grid should be full height' );
  assert.ok( a.left === 0, 'a should be left aligned in its cell' );
  assert.ok( a.top === 0, 'a should be top aligned in its cell' );
  assert.ok( LayoutTestUtils.aboutEqual( b.centerX, fullWidth / 2 ), 'b should be centered in its cell' );
  assert.ok( LayoutTestUtils.aboutEqual( b.centerY, fullHeight / 4 ), 'b should be centered in its cell' );
  assert.ok( LayoutTestUtils.aboutEqual( c.right, fullWidth ), 'c should be right aligned in its cell' );
  assert.ok( LayoutTestUtils.aboutEqual( c.bottom, fullHeight / 2 ), 'c should be bottom aligned in its cell' );
  assert.ok( LayoutTestUtils.aboutEqual( d.centerX, fullWidth / 6 ), 'd should be centered in its cell' );
  assert.ok( LayoutTestUtils.aboutEqual( d.centerY, 3 * fullHeight / 4 ), 'd should be centered in its cell' );
  assert.ok( LayoutTestUtils.aboutEqual( e.centerX, fullWidth / 2 ), 'e should be horizontally centered in its cell' );
  assert.ok( e.top === fullHeight / 2, 'e should be top aligned in its cell' );
  assert.ok( LayoutTestUtils.aboutEqual( f.centerX, 5 * fullWidth / 6 ), 'f should be centered in its cell' );
  assert.ok( LayoutTestUtils.aboutEqual( f.centerY, 3 * fullHeight / 4 ), 'f should be centered in its cell' );
} );

QUnit.test( 'Horizontal span/Vertical span', assert => {

  const [ a, c, d ] = LayoutTestUtils.createRectangles( 4 );

  // b is sizable to take up more space so that we can test horizontal/vertical span
  const b = new Rectangle( {
    sizable: true,
    localMinimumWidth: RECT_WIDTH,
    localMinimumHeight: RECT_HEIGHT
  } );
  a.layoutOptions = { column: 0, row: 0 };
  b.layoutOptions = { column: 1, row: 0, grow: 1, stretch: true };
  c.layoutOptions = { column: 0, row: 1 };
  d.layoutOptions = { column: 1, row: 1 };

  const totalWidth = 800;

  // Amount of horizontal spacing that will be distributed within each row (each row has 2 rectangles)
  const horizontalSpacing = totalWidth - 2 * RECT_WIDTH;

  const grid = new GridBox( {
    children: [ a, b, c, d ],
    preferredWidth: totalWidth
  } );

  assert.ok( a.left === 0, 'a should be at the left edge' );
  assert.ok( b.left === RECT_WIDTH, 'b should be to the right of a' );
  assert.ok( c.left === 0, 'c should be at the left edge' );
  assert.ok( d.left === c.right + horizontalSpacing / 2, 'd should be ' );
  assert.ok( d.centerX === b.centerX, 'd should be centered with b in its column' );

  // Now add verticalSpan to b - it should take up two rows
  b.layoutOptions = { column: 1, row: 0, verticalSpan: 2, stretch: true, grow: 1 };
  assert.ok( b.top === 0, 'b should be at the top edge' );
  assert.ok( b.bottom === grid.height, 'b should span 2 rows to the bottom' );
  assert.ok( d.top === RECT_HEIGHT, 'd should be in the bottom row' );
  assert.ok( d.bottom === c.bottom, 'd bottom should be aligned with c bottom' );

  // Now test horizontalSpan on b - it should take up 2 columns
  b.layoutOptions = { column: 1, row: 0, horizontalSpan: 2, stretch: true, grow: 1 };
  assert.ok( b.left === a.right, 'b should be adjacent to a' );
  assert.ok( b.width === totalWidth - RECT_WIDTH, 'b should span 2 columns' );
  assert.ok( d.centerX === horizontalSpacing / 2, 'd centered in its column (b spans 2 columns)' );
} );

QUnit.test( 'Spacing', assert => {

  //---------------------------------------------------------------------------------
  // consistent spacing
  //---------------------------------------------------------------------------------

  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );

  const grid = new GridBox( {
    spacing: 10,
    rows: [ [ a, b ], [ c, d ] ]
  } );

  const expectedWidth = 2 * RECT_WIDTH + 10;
  const expectedHeight = 2 * RECT_HEIGHT + 10;
  assert.ok( grid.width === expectedWidth, 'grid should have correct width' );
  assert.ok( grid.height === expectedHeight, 'grid should have correct height' );
  assert.ok( a.left === 0, 'a should be at the left edge' );
  assert.ok( b.left === RECT_WIDTH + 10, 'b should be to the right of a' );
  assert.ok( c.left === 0, 'c should be at the left edge' );
  assert.ok( d.left === RECT_WIDTH + 10, 'd should be to the right of c' );
  assert.ok( a.top === 0, 'a should be at the top edge' );
  assert.ok( b.top === 0, 'b should be at the top edge' );
  assert.ok( c.top === RECT_HEIGHT + 10, 'c should be below a' );
  assert.ok( d.top === RECT_HEIGHT + 10, 'd should be below b' );
} );

QUnit.test( 'margins', assert => {
  const [ a, b, c, d ] = LayoutTestUtils.createRectangles( 4 );

  const margin = 5;
  const grid = new GridBox( {
    margin: margin,
    rows: [ [ a, b ], [ c, d ] ]
  } );

  const expectedWidth = 2 * RECT_WIDTH + margin * 4;
  const expectedHeight = 2 * RECT_HEIGHT + margin * 4;

  assert.ok( grid.width === expectedWidth, 'grid should have correct width' );
  assert.ok( grid.height === expectedHeight, 'grid should have correct height' );
  assert.ok( a.left === 5, 'a should be at the left edge, with a margin' );
  assert.ok( b.left === a.right + margin * 2, 'b should be to the right of a, with a margin on each side' );
  assert.ok( c.left === 5, 'c should be at the left edge, with a margin' );

  // Add a topMargin to b - should override the grid and push the rectangle down
  const additionalMargin = 10;
  b.layoutOptions = { topMargin: additionalMargin };

  // New expected height is the margin + additionalMargin + RECT_HEIGHT + margin + RECT_HEIGHT + margin -
  // individual margins compound with grid margin
  const newExpectedHeight = margin + additionalMargin + RECT_HEIGHT + margin + RECT_HEIGHT + margin;
  assert.ok( grid.height === newExpectedHeight, 'grid should have correct height' );
} );