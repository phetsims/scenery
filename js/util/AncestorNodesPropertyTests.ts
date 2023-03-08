// Copyright 2023, University of Colorado Boulder

/**
 * QUnit tests for AncestorNodesPropertyTests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { AncestorNodesProperty, Node } from '../imports.js';

QUnit.module( 'AncestorNodesProperty' );

QUnit.test( 'AncestorNodesProperty', assert => {

  const a = new Node();
  const b = new Node();
  const c = new Node();
  const d = new Node();

  b.addChild( a );

  const ancestorNodesProperty = new AncestorNodesProperty( a );

  const checkAncestors = ( nodes: Node[], message: string ) => {
    assert.ok( ancestorNodesProperty.value.size === nodes.length, message );

    nodes.forEach( node => {
      assert.ok( ancestorNodesProperty.value.has( node ), message );
    } );
  };

  // b -> a
  checkAncestors( [ b ], 'initial' );

  // a
  b.removeChild( a );
  checkAncestors( [], 'removed from b' );

  // c -> b -> a
  c.addChild( b );
  b.addChild( a );
  checkAncestors( [ b, c ], 'added two at a time' );

  //    b
  //  /   \
  // c ->  a
  c.addChild( a );
  checkAncestors( [ b, c ], 'DAG, still the same' );

  //    b
  //  /
  // c ->  a
  b.removeChild( a );
  checkAncestors( [ c ], 'only c directly' );

  //         b
  //       /
  // d -> c ->  a
  d.addChild( c );
  checkAncestors( [ c, d ], 'added ancestor!' );

  //    b
  //     \
  // d -> c ->  a
  c.removeChild( b );
  b.addChild( c );
  checkAncestors( [ b, c, d ], 'moved b to ancestor' );

  // a
  c.removeChild( a );
  checkAncestors( [], 'nothing' );

  //    b
  //     \
  // d -> c ->  a
  c.addChild( a );
  checkAncestors( [ b, c, d ], 'back' );

  ancestorNodesProperty.dispose();
} );

