// Copyright 2021-2022, University of Colorado Boulder

/**
 * A Node that is composed with ReadingBlock.
 *
 * // REVIEW: Sounds good! That said, I see usages in JT, which is soon to be going out for production, https://github.com/phetsims/scenery/issues/1223
 * WARNING: Under active development, not ready for production. This is sort of trying out
 * inheritance for ReadingBlock/Voicing by composing into a base type to be extended by other
 * Nodes. If this is clean, maybe ReadingBlock and Voicing should be made classes rather than
 * traits.
 *
 * // REVIEW: I'm pretty confused by this type, ReadingBlock is overwriting the tagName, so what is this type doing? https://github.com/phetsims/scenery/issues/1223
 *
 * @author Jesse Greenberg
 */

import merge from '../../../../phet-core/js/merge.js';
import { scenery, Node, ReadingBlock, ReadingBlockHighlight } from '../../imports.js';

class ReadingBlockNode extends ReadingBlock( Node ) {

  /**
   * @param {Object} [options]
   */
  constructor( options ) {

    options = merge( {

      // pdom
      tagName: 'p' // typical "reading block" is just a readable paragraph in the PDOM
    }, options );

    super();

    // default highlight for a ReadingBlock is styled to indicate that the Node is different
    // from other interactive things, but is still clickable
    this.focusHighlight = options.focusHighlight || new ReadingBlockHighlight( this );

    // mutate after initialize so that we can pass through ReadingBlock options
    this.mutate( options );
  }
}

scenery.register( 'ReadingBlockNode', ReadingBlockNode );
export default ReadingBlockNode;