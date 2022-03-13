// Copyright 2021, University of Colorado Boulder

/**
 * The highlight with styling used for ReadingBlocks when the are "activated" and
 * the Voicing framework is speaking the content for a Node that composes ReadingBlock.
 *
 * @author Jesse Greenberg
 */

import merge from '../../../../phet-core/js/merge.js';
import { scenery, FocusHighlightFromNode } from '../../imports.js';

// constants
const ACTIVATED_HIGHLIGHT_COLOR = 'rgba(255,255,0,0.5)';

class ActivatedReadingBlockHighlight extends FocusHighlightFromNode {

  /**
   * @param {Node|null} node
   * @param {Object} [options]
   */
  constructor( node, options ) {

    options = merge( {
      innerStroke: null,
      outerStroke: null,
      fill: ACTIVATED_HIGHLIGHT_COLOR
    }, options );

    super( node, options );
  }
}

// @public
ActivatedReadingBlockHighlight.ACTIVATED_HIGHLIGHT_COLOR = ACTIVATED_HIGHLIGHT_COLOR;

scenery.register( 'ActivatedReadingBlockHighlight', ActivatedReadingBlockHighlight );
export default ActivatedReadingBlockHighlight;