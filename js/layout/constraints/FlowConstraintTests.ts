// Copyright 2024-2025, University of Colorado Boulder

/**
 * Unit tests for FlowConstraint.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Property from '../../../../axon/js/Property.js';
import FlowCell from '../../layout/constraints/FlowCell.js';
import FlowConstraint from '../../layout/constraints/FlowConstraint.js';
import type { FlowConstraintOptions } from '../../layout/constraints/FlowConstraint.js';
import Node from '../../nodes/Node.js';
import LayoutTestUtils from '../LayoutTestUtils.js';

QUnit.module( 'FlowConstraint' );

const createConstraint = ( constraintOptions: FlowConstraintOptions ) => {
  const [ a, b, c ] = LayoutTestUtils.createRectangles( 3 );
  const [ parent1, parent2, parent3 ] = LayoutTestUtils.createRectangles( 3 );

  parent1.addChild( a );
  parent2.addChild( b );
  parent3.addChild( c );

  const scene = new Node( { children: [ parent1, parent2, parent3 ] } );

  // parents are scaled randomly
  parent1.scale( 1.5 );
  parent2.scale( 2 );
  parent3.scale( 0.5 );

  const constraint = new FlowConstraint( scene, constraintOptions );

  constraint.insertCell( 0, new FlowCell( constraint, a, null ) );
  constraint.insertCell( 1, new FlowCell( constraint, b, null ) );
  constraint.insertCell( 2, new FlowCell( constraint, c, null ) );
  constraint.updateLayout();

  return [ a, b, c, parent1, parent2, parent3 ];
};

QUnit.test( 'Construction tests', assert => {
  const [ a, b, c ] = LayoutTestUtils.createRectangles( 3 );
  const [ parent1, parent2, parent3 ] = LayoutTestUtils.createRectangles( 3 );

  parent1.addChild( a );
  parent2.addChild( b );
  parent3.addChild( c );

  const scene = new Node( { children: [ parent1, parent2, parent3 ] } );

  const constraint = new FlowConstraint( scene );
  constraint.dispose();
  assert.ok( true, 'FlowConstraint construction test passed' );

  const constraint2 = new FlowConstraint( scene, { spacing: 10, justify: 'left', wrap: true } );
  constraint2.dispose();
  assert.ok( true, 'FlowConstraint construction test with options passed' );
} );

QUnit.test( 'Spacing in global frame', assert => {
  const [ a, b, c ] = createConstraint( { spacing: 10, align: 'top' } );

  // uniform spacing and alignment even though parents are scaled differently
  assert.equal( a.top, b.top, 'a and b are aligned to the top' );
  assert.equal( b.top, c.top, 'b and c are aligned to the top' );

  const aGlobalLeft = a.globalBounds.left;
  const aGlobalRight = a.globalBounds.right;
  const bGlobalLeft = b.globalBounds.left;
  const bGlobalRight = b.globalBounds.right;
  const cGlobalLeft = c.globalBounds.left;

  assert.ok( aGlobalLeft === 0, 'a is at the left edge of the scene' );
  assert.ok( LayoutTestUtils.aboutEqual( aGlobalRight + 10, bGlobalLeft ), 'a and b are spaced 10 apart in scene frame' );
  assert.ok( LayoutTestUtils.aboutEqual( bGlobalRight + 10, cGlobalLeft ), 'b and c are spaced 10 apart in scene frame' );
} );

QUnit.test( 'Wrap, lineSpacing', assert => {

  // A flow constraint with wrap and lineSpacing - rectangle b is tallest, so c should be wrapped below b, but
  // to the left edge of the scene.
  const lineSpacing = 10;
  const [ a, b, c ] = createConstraint( {
    preferredWidthProperty: new Property( LayoutTestUtils.RECT_WIDTH * 2 ),
    wrap: true,
    lineSpacing: lineSpacing
  } );

  const aGlobalLeft = a.globalBounds.left;
  const bGlobalBottom = b.globalBounds.bottom;
  const cGlobalLeft = c.globalBounds.left;
  const cGlobalTop = c.globalBounds.top;

  assert.ok( aGlobalLeft === 0, 'a is at the left edge of the scene' );
  assert.ok( cGlobalLeft === 0, 'c is is wrapped to the left edge of the scene' );
  assert.ok( LayoutTestUtils.aboutEqual( bGlobalBottom + lineSpacing, cGlobalTop ), 'c is wrapped below b plus lineSpacing' );
} );

QUnit.test( 'justify and justifyLines', assert => {

  // justify
  const [ a, b, c ] = createConstraint( {
    preferredWidthProperty: new Property( LayoutTestUtils.RECT_WIDTH * 3 ),
    wrap: true,
    justify: 'right',
    align: 'top'
  } );

  // a should be in the first row while be and c are in the second row, aligned to the right
  assert.ok( a.globalBounds.right === LayoutTestUtils.RECT_WIDTH * 3, 'a is at the right edge of the scene' );
  assert.ok( c.globalBounds.right === LayoutTestUtils.RECT_WIDTH * 3, 'c is at the right edge of the scene' );
  assert.ok( b.globalBounds.right === c.globalBounds.left, 'b is to the left of c' );
  assert.ok( c.globalBounds.top === b.globalBounds.top, 'c is aligned to b' );

  // justifyLines
  const [ d, e, f ] = createConstraint( {
    preferredWidthProperty: new Property( 20 ),
    preferredHeightProperty: new Property( LayoutTestUtils.RECT_HEIGHT * 4 ),
    wrap: true,
    justifyLines: 'bottom'
  } );

  // in a vertical column, left aligned
  assert.ok( d.globalBounds.left === 0, 'd is at the left edge of the scene' );
  assert.ok( e.globalBounds.left === 0, 'e is at the left edge of the scene' );
  assert.ok( f.globalBounds.left === 0, 'f is at the left edge of the scene' );

  // TODO, why doesn't this pass? https://github.com/phetsims/scenery/issues/1465
  // assert.ok( f.globalBounds.bottom === LayoutTestUtils.RECT_HEIGHT * 4, 'd is at the bottom edge of the scene (justifyLines: bottom)' );
} );