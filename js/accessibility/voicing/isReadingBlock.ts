// Copyright 2025, University of Colorado Boulder

/**
 * Checks whether something is of the ReadingBlock type.
 * Extracted to reduce circular dependencies.
 *
 * @author Jesse Greenberg
 */

import type IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import Node from '../../../../scenery/js/nodes/Node.js';
import type { ReadingBlockNode } from './ReadingBlock.js';

export function isReadingBlock( something: IntentionalAny ): something is ReadingBlockNode {
  return something instanceof Node && ( something as ReadingBlockNode )._isReadingBlock;
}