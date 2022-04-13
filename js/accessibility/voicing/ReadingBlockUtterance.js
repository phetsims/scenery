// Copyright 2021, University of Colorado Boulder

/**
 * An Utterance specifically for ReadingBlocks. When a ReadingBlock is activated, the Trail from
 * the event is used to determine exactly which ReadingBlock instance to highlight in the
 * HighlightOverlay.
 *
 * @author Jesse Greenberg
 */

import Utterance from '../../../../utterance-queue/js/Utterance.js';
import { scenery } from '../../imports.js';

class ReadingBlockUtterance extends Utterance {

  /**
   * @param {Focus|null} focus
   * @param {Object} [options]
   */
  constructor( focus, options ) {
    super( options );

    // @public {Focus|null} - Can be set and change to support reusing this ReadingBlockUtterance.
    this.readingBlockFocus = focus;
  }
}

scenery.register( 'ReadingBlockUtterance', ReadingBlockUtterance );
export default ReadingBlockUtterance;