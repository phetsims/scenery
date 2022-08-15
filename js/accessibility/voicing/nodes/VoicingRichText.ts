// Copyright 2021-2022, University of Colorado Boulder

/**
 * RichText that composes ReadingBlock, adding support for Voicing and input listeners that speak content upon
 * user activation.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import IProperty from '../../../../../axon/js/IProperty.js';
import optionize, { combineOptions, EmptySelfOptions } from '../../../../../phet-core/js/optionize.js';
import { ReadingBlock, ReadingBlockHighlight, ReadingBlockOptions, RichText, RichTextOptions, scenery } from '../../../imports.js';

type SelfOptions = EmptySelfOptions;
type ParentOptions = ReadingBlockOptions & RichTextOptions;
export type VoicingRichTextOptions = SelfOptions & ParentOptions;

class VoicingRichText extends ReadingBlock( RichText ) {

  public constructor( text: string | IProperty<string>, providedOptions?: VoicingRichTextOptions ) {

    const initialText = typeof text === 'string' ? text : text.value;

    let options = optionize<VoicingRichTextOptions, SelfOptions, ParentOptions>()( {

      // {string|null} - if provided, alternative text that will be read that is different from the
      // visually displayed text
      readingBlockNameResponse: null,

      // pdom
      innerContent: initialText,

      // voicing
      // default tag name for a ReadingBlock, but there are cases where you may want to override this (such as
      // RichText links)
      readingBlockTagName: 'button'
    }, providedOptions );

    // Options that use other options
    // @ts-ignore
    options = combineOptions<VoicingRichTextOptions>( options, {
      readingBlockNameResponse: options.readingBlockNameResponse || initialText
    } );

    super( initialText );

    this.focusHighlight = new ReadingBlockHighlight( this );

    this.mutate( options );

    if ( typeof text !== 'string' ) {
      this.textProperty = text;

      // TODO: We might be memory leaking here, we'll want to dispose for https://github.com/phetsims/chipper/issues/1302
      text.link( string => {
        this.innerContent = string;
        this.readingBlockNameResponse = options.readingBlockHintResponse || string;
      } );
    }
  }
}

scenery.register( 'VoicingRichText', VoicingRichText );
export default VoicingRichText;
