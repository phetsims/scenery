// Copyright 2024-2025, University of Colorado Boulder

/**
 * A utility function that enforces that no additional Children are added to a Node.
 *
 * In particular, useful for making sure that Nodes are not decorated with other content which can be
 * problematic for dynamic layout. See https://github.com/phetsims/sun/issues/860.
 *
 * Usage:
 *
 * const myNode = new Node();
 * myNode.children = [ new Node(), new Node() ]; // fill in with your own content
 * assertNoAdditionalChildren( myNode ); // prevent new children
 *
 * Note that removals need to be allowed for disposal.
 *
 * @author Jesse Greenberg
 */

import Node from '../nodes/Node.js';

/**
 * @param node - Prevent changes on this Node
 */
const assertNoAdditionalChildren = function( node: Node ): void {
  if ( assert ) {

    node.insertChild = function( index: number, node: Node, isComposite?: boolean ): Node {
      assert && assert( false, 'Attempt to insert child into Leaf' );
      return node;
    };
  }
};

export default assertNoAdditionalChildren;