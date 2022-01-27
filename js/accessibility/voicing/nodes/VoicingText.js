// Copyright 2021-2022, University of Colorado Boulder

/**
 * Text that mixes ReadingBlock, supporting features of Voicing and adding listeners that speak the text string
 * upon user input.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import merge from '../../../../../phet-core/js/merge.js';
import { ReadingBlock, ReadingBlockHighlight, scenery, Text } from '../../../imports.js';

class VoicingText extends ReadingBlock( Text ) {

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

    // Options that use other options
    options = merge( options, {
      readingBlockContent: options.readingBlockContent || text
    } );

    super( text );

    // unique highlight for non-interactive components
    this.focusHighlight = new ReadingBlockHighlight( this );

    this.mutate( options );
  }
}

scenery.register( 'VoicingText', VoicingText );
export default VoicingText;
