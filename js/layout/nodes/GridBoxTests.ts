// Copyright 2024-2025, University of Colorado Boulder

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
 *  - pixel comparison tests
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

// I want to use spread to work with options easily in this testing file.
/* eslint-disable phet/no-object-spread-on-non-literals */

import Rectangle from '../../nodes/Rectangle.js';
import PixelComparisonTestUtils from '../../tests/PixelComparisonTestUtils.js';
import LayoutTestUtils from '../LayoutTestUtils.js';
import GridBox from '../nodes/GridBox.js';

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


const DEFAULT_THRESHOLD = PixelComparisonTestUtils.DEFAULT_THRESHOLD;
const testedRenderers = PixelComparisonTestUtils.TESTED_RENDERERS;

if ( PixelComparisonTestUtils.platformSupportsPixelComparisonTests() ) {

  // Options reused for all tests involving a top row of rectangles that resize based on contents in the column.
  const topOptions = {
    sizable: true
  };
  const topLayoutOptions = {
    row: 0,
    minContentWidth: 50,
    minContentHeight: 50
  };

  //---------------------------------------------------------------------------------
  // Pixel Comparison: Simple GridConstraint
  //---------------------------------------------------------------------------------
  const simpleGridConstraintUrl = 'data:image/png;base64,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';
  PixelComparisonTestUtils.multipleRendererTest( 'Simple GridBox',
    ( scene, display ) => {
      display.width = 400;
      display.height = 400;

      const box = new GridBox( {
        stretch: true,
        children: [
          new Rectangle( {
            fill: 'red',
            layoutOptions: { column: 0, grow: 1, maxContentWidth: 100, ...topLayoutOptions }, ...topOptions
          } ),
          new Rectangle( { fill: 'orange', layoutOptions: { column: 1, grow: 1, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'yellow', layoutOptions: { column: 2, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'green', layoutOptions: { column: 3, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'blue', layoutOptions: { column: 4, ...topLayoutOptions }, ...topOptions } )
        ]
      } );

      scene.addChild( box );

      display.updateDisplay();
    }, simpleGridConstraintUrl,
    DEFAULT_THRESHOLD, testedRenderers
  );

  //---------------------------------------------------------------------------------
  // Pixel Comparison: GridConstraint with width limits
  //---------------------------------------------------------------------------------
  const growWithMinMaxWidthsUrl = 'data:image/png;base64,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020IECBAgACBAAEFKyAEIxAgQIAAAQK7BBSsXXm6DQECBAgQIBAgoGAFhGAEAgQIECBAYJeAgrUrT7chQIAAAQIEAgQUrIAQjECAAAECBAjsElCwduXpNgQIECBAgECAgIIVEIIRCBAgQIAAgV0CCtauPN2GAAECBAgQCBBQsAJCMAIBAgQIECCwS0DB2pWn2xAgQIAAAQIBAgpWQAhGIECAAAECBHYJKFi78nQbAgQIECBAIEBAwQoIwQgECBAgQIDALgEFa1eebkOAAAECBAgECChYASEYgQABAgQIENgloGDtytNtCBAgQIAAgQABBSsgBCMQIECAAAECuwQUrF15ug0BAgQIECAQIKBgBYRgBAIECBAgQGCXgIK1K0+3IUCAAAECBAIEFKyAEIxAgAABAgQI7BJQsHbl6TYECBAgQIBAgICCFRCCEQgQIECAAIFdAgrWrjzdhgABAgQIEAgQULACQjACAQIECBAgsEtAwdqVp9sQIECAAAECAQIKVkAIRiBAgAABAgR2CShYu/J0GwIECBAgQCBAQMEKCMEIBAgQIECAwC4BBWtXnm5DgAABAgQIBAgoWAEhGIEAAQIECBDYJaBg7crTbQgQIECAAIEAAQUrIAQjECBAgAABArsEFKxdeboNAQIECBAgECCgYAWEYAQCBAgQIEBgl4CCtStPtyFAgAABAgQCBBSsgBCMQIAAAQIECOwSULB25ek2BAgQIECAQICAghUQghEIECBAgACBXQIK1q483YYAAQIECBAIEFCwAkIwAgECBAgQILBLQMHalafbECBAgAABAgECClZACEYgQIAAAQIEdgkoWLvydBsCBAgQIEAgQEDBCgjBCAQIECBAgMAuAQVrV55uQ4AAAQIECAQIKFgBIRiBAAECBAgQ2CWgYO3K020IECBAgACBAAEFKyAEIxAgQIAAAQK7BBSsXXm6DQECBAgQIBAgoGAFhGAEAgQIECBAYJeAgrUrT7chQIAAAQIEAgQUrIAQjECAAAECBAjsElCwduXpNgQIECBAgECAgIIVEIIRCBAgQIAAgV0CCtauPN2GAAECBAgQCBBQsAJCMAIBAgQIECCwS0DB2pWn2xAgQIAAAQIBAgpWQAhGIECAAAECBHYJKFi78nQbAgQIECBAIEBAwQoIwQgECBAgQIDALgEFa1eebkOAAAECBAgECChYASEYgQABAgQIENgloGDtytNtCBAgQIAAgQABBSsgBCMQIECAAAECuwQUrF15ug0BAgQIECAQIKBgBYRgBAIECBAgQGCXgIK1K0+3IUCAAAECBAIEFKyAEIxAgAABAgQI7BJQsHbl6TYECBAgQIBAgICCFRCCEQgQIECAAIFdAgrWrjzdhgABAgQIEAgQULACQjACAQIECBAgsEtAwdqVp9sQIECAAAECAQIKVkAIRiBAgAABAgR2CShYu/J0GwIECBAgQCBAQMEKCMEIBAgQIECAwC4BBWtXnm5DgAABAgQIBAgoWAEhGIEAAQIECBDYJaBg7crTbQgQIECAAIEAAQUrIAQjECBAgAABArsEFKxdeboNAQIECBAgECCgYAWEYAQCBAgQIEBgl4CCtStPtyFAgAABAgQCBBSsgBCMQIAAAQIECOwSULB25ek2BAgQIECAQICAghUQghEIECBAgACBXQIK1q483YYAAQIECBAIEFCwAkIwAgECBAgQILBLQMHalafbECBAgAABAgECClZACEYgQIAAAQIEdgkoWLvydBsCBAgQIEAgQEDBCgjBCAQIECBAgMAuAQVrV55uQ4AAAQIECAQIKFgBIRiBAAECBAgQ2CXwX2GoJ0v/GLPWAAAAAElFTkSuQmCC';
  PixelComparisonTestUtils.multipleRendererTest( 'Grow with min/max width constraints',
    ( scene, display ) => {
      display.width = 600;
      display.height = 300;

      const box = new GridBox( {
        stretch: true,
        children: [
          new Rectangle( {
            fill: 'red',
            layoutOptions: { column: 0, grow: 1, maxContentWidth: 100, ...topLayoutOptions }, ...topOptions
          } ),
          new Rectangle( { fill: 'orange', layoutOptions: { column: 1, grow: 1, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'yellow', layoutOptions: { column: 2, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'green', layoutOptions: { column: 3, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'blue', layoutOptions: { column: 4, ...topLayoutOptions }, ...topOptions } ),

          new Rectangle( 0, 0, 350, 50, { fill: 'black', layoutOptions: { row: 1, column: 0, horizontalSpan: 2 } } ),
          new Rectangle( 0, 0, 300, 50, { fill: 'black', layoutOptions: { row: 2, column: 1, horizontalSpan: 2 } } ),
          new Rectangle( 0, 0, 300, 50, { fill: 'black', layoutOptions: { row: 3, column: 1, horizontalSpan: 2 } } )
        ]
      } );

      scene.addChild( box );

      display.updateDisplay();
    }, growWithMinMaxWidthsUrl,
    DEFAULT_THRESHOLD, testedRenderers
  );

  //---------------------------------------------------------------------------------
  // Proportional Grow
  //---------------------------------------------------------------------------------

  const proportionalGrowUrl = 'data:image/png;base64,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K2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKmBglRYnNgECBAgQIJArYGDldiMZAQIECBAgUCpgYJUWJzYBAgQIECCQK2Bg5XYjGQECBAgQIFAqYGCVFic2AQIECBAgkCtgYOV2IxkBAgQIECBQKvAXk98tPGL5Ma0AAAAASUVORK5CYII=';
  PixelComparisonTestUtils.multipleRendererTest( 'Constraint with proportional grow',
    ( scene, display ) => {
      const box = new GridBox( {
        stretch: true,
        children: [
          new Rectangle( { fill: 'red', layoutOptions: { column: 0, grow: 2, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'orange', layoutOptions: { column: 1, grow: 3, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'yellow', layoutOptions: { column: 2, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'green', layoutOptions: { column: 3, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'blue', layoutOptions: { column: 4, ...topLayoutOptions }, ...topOptions } ),

          new Rectangle( 0, 0, 350, 50, { fill: 'black', layoutOptions: { row: 1, column: 0, horizontalSpan: 3 } } )
        ]
      } );

      scene.addChild( box );

      display.width = 600;
      display.height = 300;
      display.updateDisplay();
    }, proportionalGrowUrl,
    DEFAULT_THRESHOLD, testedRenderers
  );

  //---------------------------------------------------------------------------------
  // Default (distributed) grow
  //---------------------------------------------------------------------------------
  const defaultGrowUrl = 'data:image/png;base64,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CBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVIBgVU6nLMJECBAgACBXAGBlbuNywgQIECAAIFSAYFVOpyzCRAgQIAAgVwBgZW7jcsIECBAgACBUgGBVTqcswkQIECAAIFcAYGVu43LCBAgQIAAgVKBvwEZsJE87J6RIAAAAABJRU5ErkJggg==';
  PixelComparisonTestUtils.multipleRendererTest( 'Default grow',
    ( scene, display ) => {

      const box = new GridBox( {
        stretch: true,
        children: [
          new Rectangle( { fill: 'red', layoutOptions: { column: 0, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'orange', layoutOptions: { column: 1, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'yellow', layoutOptions: { column: 2, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'green', layoutOptions: { column: 3, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'blue', layoutOptions: { column: 4, ...topLayoutOptions }, ...topOptions } ),

          new Rectangle( 0, 0, 350, 50, { fill: 'black', layoutOptions: { row: 1, column: 0, horizontalSpan: 3 } } )
        ]
      } );

      scene.addChild( box );

      display.width = 600;
      display.height = 300;
      display.updateDisplay();
    }, defaultGrowUrl,
    DEFAULT_THRESHOLD, testedRenderers
  );

  //---------------------------------------------------------------------------------
  // Max constrained grow
  //---------------------------------------------------------------------------------
  const maxConstrainedGrowUrl = 'data:image/png;base64,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AAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBUQWKPDOZsAAQIECBDoCgis7jYuI0CAAAECBEYFBNbocM4mQIAAAQIEugICq7uNywgQIECAAIFRAYE1OpyzCRAgQIAAga6AwOpu4zICBAgQIEBgVEBgjQ7nbAIECBAgQKArILC627iMAAECBAgQGBX4DyjULTw4Yw+SAAAAAElFTkSuQmCC';
  PixelComparisonTestUtils.multipleRendererTest( 'Max Constrained Grow',
    ( scene, display ) => {
      const box = new GridBox( {
        stretch: true,
        children: [
          new Rectangle( { fill: 'red', layoutOptions: { column: 0, maxContentWidth: 100, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'orange', layoutOptions: { column: 1, maxContentWidth: 100, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'yellow', layoutOptions: { column: 2, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'green', layoutOptions: { column: 3, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'blue', layoutOptions: { column: 4, ...topLayoutOptions }, ...topOptions } ),

          new Rectangle( 0, 0, 350, 50, { fill: 'black', layoutOptions: { row: 1, column: 0, horizontalSpan: 3 } } )
        ]
      } );

      scene.addChild( box );

      display.width = 600;
      display.height = 300;
      display.updateDisplay();
    }, maxConstrainedGrowUrl,
    DEFAULT_THRESHOLD, testedRenderers
  );

  //---------------------------------------------------------------------------------
  // Unbiased grow (cells grow in a reasonable way without bias to any direction)
  //---------------------------------------------------------------------------------
  const unbiasedGrowUrl = 'data:image/png;base64,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';
  PixelComparisonTestUtils.multipleRendererTest( 'Unbiased grow',
    ( scene, display ) => {

      const box = new GridBox( {
        stretch: true,
        children: [
          new Rectangle( { fill: 'red', layoutOptions: { column: 0, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'orange', layoutOptions: { column: 1, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'yellow', layoutOptions: { column: 2, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'green', layoutOptions: { column: 3, ...topLayoutOptions }, ...topOptions } ),
          new Rectangle( { fill: 'blue', layoutOptions: { column: 4, ...topLayoutOptions }, ...topOptions } ),

          new Rectangle( 0, 0, 350, 50, { fill: 'black', layoutOptions: { row: 1, column: 0, horizontalSpan: 3 } } ),
          new Rectangle( 0, 0, 350, 50, { fill: 'black', layoutOptions: { row: 2, column: 1, horizontalSpan: 3 } } ),
          new Rectangle( 0, 0, 350, 50, { fill: 'black', layoutOptions: { row: 3, column: 2, horizontalSpan: 3 } } )
        ]
      } );

      scene.addChild( box );

      display.width = 600;
      display.height = 300;
      display.updateDisplay();
    }, unbiasedGrowUrl,
    DEFAULT_THRESHOLD, testedRenderers
  );
}