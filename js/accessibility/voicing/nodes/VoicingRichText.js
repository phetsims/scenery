// Copyright 2021-2022, University of Colorado Boulder

/**
 * RichText that composes ReadingBlock, adding support for Voicing and input listeners that speak content upon
 * user activation.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import merge from '../../../../../phet-core/js/merge.js';
import { ReadingBlock, ReadingBlockHighlight, RichText, scenery } from '../../../imports.js';

class VoicingRichText extends ReadingBlock( RichText, 1 ) {

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

    // Options that use other options
    options = merge( options, {
      readingBlockContent: options.readingBlockContent || text
    } );

    super( text );

    this.focusHighlight = new ReadingBlockHighlight( this );

    this.mutate( options );
  }
}

scenery.register( 'VoicingRichText', VoicingRichText );
export default VoicingRichText;
