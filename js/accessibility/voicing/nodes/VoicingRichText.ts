// Copyright 2021-2022, University of Colorado Boulder

/**
 * RichText that composes ReadingBlock, adding support for Voicing and input listeners that speak content upon
 * user activation.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TReadOnlyProperty from '../../../../../axon/js/TReadOnlyProperty.js';
import optionize, { EmptySelfOptions } from '../../../../../phet-core/js/optionize.js';
import { ReadingBlock, ReadingBlockHighlight, ReadingBlockOptions, RichText, RichTextOptions, scenery } from '../../../imports.js';

type SelfOptions = EmptySelfOptions;
type ParentOptions = ReadingBlockOptions & RichTextOptions;
export type VoicingRichTextOptions = SelfOptions & ParentOptions;

class VoicingRichText extends ReadingBlock( RichText ) {

  public constructor( text: string | TReadOnlyProperty<string>, providedOptions?: VoicingRichTextOptions ) {

    const options = optionize<VoicingRichTextOptions, SelfOptions, ParentOptions>()( {

      // {string|null} - if provided, alternative text that will be read that is different from the
      // visually displayed text
      readingBlockNameResponse: text,

      // pdom
      innerContent: text,

      // voicing
      // default tag name for a ReadingBlock, but there are cases where you may want to override this (such as
      // RichText links)
      readingBlockTagName: 'button'
    }, providedOptions );

    super( text );

    this.focusHighlight = new ReadingBlockHighlight( this );

    this.mutate( options );
  }
}

scenery.register( 'VoicingRichText', VoicingRichText );
export default VoicingRichText;
