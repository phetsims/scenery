// Copyright 2021, University of Colorado Boulder

/**
 * Text that mixes ReadingBlock, supporting features of Voicing and adding listeners that speak the text string
 * upon user input.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import merge from '../../../../../phet-core/js/merge.js';
import Text from '../../../nodes/Text.js';
import scenery from '../../../scenery.js';
import ReadingBlock from '../ReadingBlock.js';
import ReadingBlockHighlight from '../ReadingBlockHighlight.js';

class VoicingText extends Text {

  /**
   * @param {string} text
   * @param {Object} [options]
   */
  constructor( text, options ) {
    options = merge( {

      // {string|null} - if provided, alternative text that will be spoken that is different from the
      // visually displayed text
      readingBlockContent: null,

      // pdom
      tagName: 'p',
      innerContent: text
    }, options );

    super( text, options );

    // unique highlight for non-interactive components
    this.focusHighlight = new ReadingBlockHighlight( this );

    // voicing
    this.initializeReadingBlock( {
      readingBlockContent: options.readingBlockContent || text
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

ReadingBlock.compose( VoicingText );

scenery.register( 'VoicingText', VoicingText );
export default VoicingText;
