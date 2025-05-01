// Copyright 2021-2025, University of Colorado Boulder

/**
 * RichText that composes ReadingBlock, adding support for Voicing and input listeners that speak content upon
 * user activation.
 *
 * Example usage:
 *   const voicingRichText = new VoicingRichText( 'Hello, world!' );
 *
 * Example usage:
 *   const voicingRichText = new VoicingRichText( 'Hello, world!', {
 *     accessibleParagraph: 'Custom Voicing Text'
 *   } );
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TReadOnlyProperty from '../../../../../axon/js/TReadOnlyProperty.js';
import optionize, { EmptySelfOptions } from '../../../../../phet-core/js/optionize.js';
import StrictOmit from '../../../../../phet-core/js/types/StrictOmit.js';
import type { ReadingBlockOptions } from '../../../accessibility/voicing/ReadingBlock.js';
import ReadingBlock from '../../../accessibility/voicing/ReadingBlock.js';
import ReadingBlockHighlight from '../../../accessibility/voicing/ReadingBlockHighlight.js';
import type { RichTextOptions } from '../../../nodes/RichText.js';
import RichText from '../../../nodes/RichText.js';
import scenery from '../../../scenery.js';

type SelfOptions = EmptySelfOptions;

// focusHighlight will always be set by this class
type ParentOptions = ReadingBlockOptions & StrictOmit<RichTextOptions, 'focusHighlight' | 'innerContent'>;
export type VoicingRichTextOptions = SelfOptions & ParentOptions;

class VoicingRichText extends ReadingBlock( RichText ) {

  public constructor( text: string | TReadOnlyProperty<string>, providedOptions?: VoicingRichTextOptions ) {
    const options = optionize<VoicingRichTextOptions, SelfOptions, ParentOptions>()( {

      // {string|null} - if provided, alternative text that will be read that is different from the
      // visually displayed text
      readingBlockNameResponse: text,

      // voicing
      // default tag name for a ReadingBlock, but there are cases where you may want to override this (such as
      // RichText links)
      readingBlockTagName: 'button'
    }, providedOptions );


    // Use accessibleParagraph so that the RichText will appear in the PDOM as a paragraph.
    options.accessibleParagraph = options.accessibleParagraph === undefined ? text : options.accessibleParagraph;

    super( text, options );

    this.focusHighlight = new ReadingBlockHighlight( this );
  }
}

scenery.register( 'VoicingRichText', VoicingRichText );
export default VoicingRichText;