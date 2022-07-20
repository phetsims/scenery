// Copyright 2021-2022, University of Colorado Boulder

/**
 * Text that mixes ReadingBlock, supporting features of Voicing and adding listeners that speak the text string
 * upon user input.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import optionize, { combineOptions } from '../../../../../phet-core/js/optionize.js';
import { EmptySelfOptions } from '../../../../../phet-core/js/optionize.js';
import { ReadingBlock, ReadingBlockHighlight, ReadingBlockOptions, scenery, Text, TextOptions } from '../../../imports.js';

type SelfOptions = EmptySelfOptions;
type ParentOptions = ReadingBlockOptions & TextOptions;
type VoicingTextOptions = SelfOptions & ParentOptions;

class VoicingText extends ReadingBlock( Text, 1 ) {

  public constructor( text: string, providedOptions?: VoicingTextOptions ) {
    let options = optionize<VoicingTextOptions, SelfOptions, ParentOptions>()( {

      // {string|null} - if provided, alternative text that will be spoken that is different from the
      // visually displayed text
      readingBlockNameResponse: null,

      // pdom
      tagName: 'p',
      innerContent: text
    }, providedOptions );

    // Options that use other options, should be the same type as options
    options = combineOptions<typeof options>( options, {
      readingBlockNameResponse: options.readingBlockNameResponse || text
    } );

    super( text );

    // unique highlight for non-interactive components
    this.focusHighlight = new ReadingBlockHighlight( this );

    this.mutate( options );
  }
}

scenery.register( 'VoicingText', VoicingText );
export default VoicingText;
