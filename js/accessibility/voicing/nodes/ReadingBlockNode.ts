// Copyright 2025, University of Colorado Boulder

/**
 * A superclass Node that is composed with ReadingBlock, for where inheritance is a preferred pattern.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import { EmptySelfOptions } from '../../../../../phet-core/js/optionize.js';
import StrictOmit from '../../../../../phet-core/js/types/StrictOmit.js';
import type { ReadingBlockOptions } from '../../../accessibility/voicing/ReadingBlock.js';
import ReadingBlock from '../../../accessibility/voicing/ReadingBlock.js';
import Node, { NodeOptions } from '../../../nodes/Node.js';
import scenery from '../../../scenery.js';

type SelfOptions = EmptySelfOptions;
type ParentOptions = ReadingBlockOptions & StrictOmit<NodeOptions, 'tagName' | 'focusable'>;
export type ReadingBlockNodeOptions = SelfOptions & ParentOptions;

class ReadingBlockNode extends ReadingBlock( Node ) {
  public constructor( providedOptions?: ReadingBlockNodeOptions ) {
    super( providedOptions );
  }
}

scenery.register( 'ReadingBlockNode', ReadingBlockNode );
export default ReadingBlockNode;