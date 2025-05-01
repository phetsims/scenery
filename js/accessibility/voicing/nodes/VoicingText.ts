// Copyright 2021-2025, University of Colorado Boulder

/**
 * Text that mixes ReadingBlock, supporting features of Voicing and adding listeners that speak the text string
 * upon user input.
 *
 * Example usage:
 *   const voicingText = new VoicingText( 'Hello, world!' );
 *
 * Example usage:
 *   const voicingText = new VoicingText( 'Hello, world!', {
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
import type { TextOptions } from '../../../nodes/Text.js';
import Text from '../../../nodes/Text.js';
import scenery from '../../../scenery.js';

type SelfOptions = EmptySelfOptions;
type ParentOptions = ReadingBlockOptions & StrictOmit<TextOptions, 'innerContent' | 'focusHighlight'>;
export type VoicingTextOptions = SelfOptions & ParentOptions;

class VoicingText extends ReadingBlock( Text ) {

  public constructor( text: string | TReadOnlyProperty<string>, providedOptions?: VoicingTextOptions ) {
    const options = optionize<VoicingTextOptions, SelfOptions, ParentOptions>()( {

      // {string|null} - if provided, alternative text that will be spoken that is different from the
      // visually displayed text
      readingBlockNameResponse: text
    }, providedOptions );


    // Use accessibleParagraph so that the RichText will appear in the PDOM as a paragraph.
    options.accessibleParagraph = options.accessibleParagraph === undefined ? text : options.accessibleParagraph;

    super( text );

    // unique highlight for non-interactive components
    this.focusHighlight = new ReadingBlockHighlight( this );

    this.mutate( options );
  }
}

scenery.register( 'VoicingText', VoicingText );
export default VoicingText;