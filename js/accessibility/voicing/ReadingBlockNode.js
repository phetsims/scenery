// Copyright 2021, University of Colorado Boulder

/**
 * A Node that is composed with ReadingBlock.
 *
 * WARNING: Under active development, not ready for production. This is sort of trying out
 * inheritance for ReadingBlock/Voicing by composing into a base type to be extended by other
 * Nodes. If this is clean, maybe ReadingBlock and Voicing should be made classes rather than
 * traits.
 *
 * @author Jesse Greenberg
 */

import merge from '../../../../phet-core/js/merge.js';
import VoicingHighlight from '../../../../scenery-phet/js/accessibility/speaker/VoicingHighlight.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import ReadingBlock from './ReadingBlock.js';

class ReadingBlockNode extends Node {

  /**
   * @param {Object} [options]
   */
  constructor( options ) {

    options = merge( {

      // pdom
      tagName: 'p' // typical "reading block" is just a readable paragraph in the PDOM
    }, options );

    super();
    this.initializeReadingBlock();

    // default highlight for a ReadingBlock is styled to indicate that the Node is different
    // from other interactive things, but is still clickable
    this.focusHighlight = options.focusHighlight || new VoicingHighlight( this );

    // mutate after initialize so that we can pass through ReadingBlock options
    this.mutate( options );
  }
}

ReadingBlock.compose( ReadingBlockNode );

scenery.register( 'ReadingBlockNode', ReadingBlockNode );
export default ReadingBlockNode;