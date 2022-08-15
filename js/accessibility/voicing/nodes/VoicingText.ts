// Copyright 2021-2022, University of Colorado Boulder

/**
 * Text that mixes ReadingBlock, supporting features of Voicing and adding listeners that speak the text string
 * upon user input.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TReadOnlyProperty from '../../../../../axon/js/TReadOnlyProperty.js';
import optionize, { combineOptions, EmptySelfOptions } from '../../../../../phet-core/js/optionize.js';
import { ReadingBlock, ReadingBlockHighlight, ReadingBlockOptions, scenery, Text, TextOptions } from '../../../imports.js';

type SelfOptions = EmptySelfOptions;
type ParentOptions = ReadingBlockOptions & TextOptions;
type VoicingTextOptions = SelfOptions & ParentOptions;

class VoicingText extends ReadingBlock( Text ) {

  public constructor( text: string | TReadOnlyProperty<string>, providedOptions?: VoicingTextOptions ) {

    const initialText = typeof text === 'string' ? text : text.value;

    let options = optionize<VoicingTextOptions, SelfOptions, ParentOptions>()( {

      // {string|null} - if provided, alternative text that will be spoken that is different from the
      // visually displayed text
      readingBlockNameResponse: null,

      // pdom
      tagName: 'p',
      innerContent: initialText
    }, providedOptions );

    // Options that use other options, should be the same type as options
    options = combineOptions<typeof options>( options, {
      readingBlockNameResponse: options.readingBlockNameResponse || initialText
    } );

    super( initialText );

    // unique highlight for non-interactive components
    this.focusHighlight = new ReadingBlockHighlight( this );

    this.mutate( options );

    if ( typeof text !== 'string' ) {
      this.mutate( {
        textProperty: text
      } );

      // TODO: We might be memory leaking here, we'll want to dispose for https://github.com/phetsims/chipper/issues/1302
      text.link( string => {
        this.innerContent = string;
        this.readingBlockNameResponse = options.readingBlockHintResponse || string;
      } );
    }
  }
}

scenery.register( 'VoicingText', VoicingText );
export default VoicingText;
