// Copyright 2025, University of Colorado Boulder

/**
 * Given a Node, search for a stringProperty in the Node or its children, recursively. This
 * is useful for finding a string to set as ParallelDOM content.
 *
 * This uses a depth first search to find the first instance of Text or RichText under the Node.
 * It won't necessarily be the closest to the root of the Node or most "prominent" Text/RichText
 * if there are multiple Text/RichText nodes.
 *
 * @author Jesse Greenberg
 */

import TReadOnlyProperty from '../../../../axon/js/TReadOnlyProperty.js';
import Node from '../../nodes/Node.js';
import RichText from '../../nodes/RichText.js';
import Text from '../../nodes/Text.js';

export const findStringProperty = ( node: Node ): TReadOnlyProperty<string> | null => {

  // Check if the node is an instance of Text or RichText and return the stringProperty
  if ( node instanceof Text || node instanceof RichText ) {
    return node.stringProperty;
  }

  // If the node has children, iterate over them recursively
  if ( node.children ) {
    for ( const child of node.children ) {
      const text = findStringProperty( child );
      if ( text ) {
        return text;
      }
    }
  }

  // Return null if text is not found
  return null;
};