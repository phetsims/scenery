// Copyright 2021-2022, University of Colorado Boulder

/**
 * An Utterance specifically for ReadingBlocks. When a ReadingBlock is activated, the Trail from
 * the event is used to determine exactly which ReadingBlock instance to highlight in the
 * HighlightOverlay.
 *
 * @author Jesse Greenberg
 */

import Utterance, { UtteranceOptions } from '../../../../utterance-queue/js/Utterance.js';
import { Focus, scenery } from '../../imports.js';

export type ReadingBlockUtteranceOptions = UtteranceOptions;

class ReadingBlockUtterance extends Utterance {

  // Can be set and change to support reusing this ReadingBlockUtterance.
  public readingBlockFocus: Focus | null;

  public constructor( focus: Focus | null, options?: ReadingBlockUtteranceOptions ) {
    super( options );
    this.readingBlockFocus = focus;
  }
}

scenery.register( 'ReadingBlockUtterance', ReadingBlockUtterance );
export default ReadingBlockUtterance;