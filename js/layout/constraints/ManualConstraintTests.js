// Copyright 2021-2022, University of Colorado Boulder

/**
 * ManualConstraint tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Node from '../../nodes/Node.js';
import Rectangle from '../../nodes/Rectangle.js';
import ManualConstraint from './ManualConstraint.js';

QUnit.module( 'ManualConstraint' );

QUnit.test( 'Identity', assert => {
  const a = new Rectangle( 0, 0, 100, 50, { fill: 'red' } );
  const b = new Rectangle( 0, 0, 100, 50, { fill: 'blue' } );
  const aContainer = new Node( { children: [ a ] } );
  const bContainer = new Node( { children: [ b ] } );
  const root = new Node( { children: [ aContainer, bContainer ] } );

  ManualConstraint.create( root, [ a, b ], ( aProxy, bProxy ) => {
    bProxy.left = aProxy.right;
  } );

  root.validateBounds();
  assert.equal( b.x, 100, 'x' );

  a.x = 100;

  root.validateBounds();
  assert.equal( b.x, 200, 'x after 100' );
} );

QUnit.test( 'Translation', assert => {
  const a = new Rectangle( 0, 0, 100, 50, { fill: 'red' } );
  const b = new Rectangle( 0, 0, 100, 50, { fill: 'blue' } );
  const aContainer = new Node( { children: [ a ] } );
  const bContainer = new Node( { children: [ b ], x: 50 } );
  const root = new Node( { children: [ aContainer, bContainer ] } );

  ManualConstraint.create( root, [ a, b ], ( aProxy, bProxy ) => {
    bProxy.left = aProxy.right;
  } );

  root.validateBounds();
  assert.equal( b.x, 50, 'x' );

  a.x = 100;

  root.validateBounds();
  assert.equal( b.x, 150, 'x after 100' );
} );

QUnit.test( 'Scale', assert => {
  const a = new Rectangle( 0, 0, 100, 50, { fill: 'red' } );
  const b = new Rectangle( 0, 0, 100, 50, { fill: 'blue' } );
  const aContainer = new Node( { children: [ a ], scale: 2 } );
  const bContainer = new Node( { children: [ b ] } );
  const root = new Node( { children: [ aContainer, bContainer ] } );

  ManualConstraint.create( root, [ a, b ], ( aProxy, bProxy ) => {
    bProxy.left = aProxy.right;
  } );

  root.validateBounds();
  assert.equal( b.x, 200, 'x' );

  a.x = 100;

  root.validateBounds();
  assert.equal( b.x, 400, 'x after 100' );
} );
