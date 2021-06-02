// Copyright 2021, University of Colorado Boulder

/**
 * RichText that composes ReadingBlock, adding support for Voicing and input listeners that speak content upon
 * user activation.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import merge from '../../../../../phet-core/js/merge.js';
import RichText from '../../../nodes/RichText.js';
import scenery from '../../../scenery.js';
import ReadingBlock from '../ReadingBlock.js';
import ReadingBlockHighlight from '../ReadingBlockHighlight.js';

class VoicingRichText extends RichText {

  /**
   * @param {string} text
   * @param {Object} [options]
   */
  constructor( text, options ) {
    options = merge( {

      // {string|null} - if provided, alternative text that will be read that is different from the
      // visually displayed text
      readingBlockContent: null,

      // pdom
      innerContent: text,

      // voicing
      // default tag name for a ReadingBlock, but there are cases where you may want to override this (such as
      // RichText links)
      readingBlockTagName: 'button'
    }, options );

    super( text, options );

    this.focusHighlight = new ReadingBlockHighlight( this );

    this.initializeReadingBlock( {
      readingBlockContent: options.readingBlockContent || text,
      readingBlockTagName: options.readingBlockTagName
    } );
  }

  /**
   * @public
   */
  dispose() {
    this.disposeReadingBlock();
    super.dispose();
  }
}

ReadingBlock.compose( VoicingRichText );

scenery.register( 'VoicingRichText', VoicingRichText );
export default VoicingRichText;
