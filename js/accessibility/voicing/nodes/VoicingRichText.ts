// Copyright 2021-2022, University of Colorado Boulder

/**
 * RichText that composes ReadingBlock, adding support for Voicing and input listeners that speak content upon
 * user activation.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import merge from '../../../../../phet-core/js/merge.js';
import optionize from '../../../../../phet-core/js/optionize.js';
import EmptyObjectType from '../../../../../phet-core/js/types/EmptyObjectType.js';
import { ReadingBlock, ReadingBlockHighlight, ReadingBlockOptions, RichText, RichTextOptions, scenery } from '../../../imports.js';

type SelfOptions = EmptyObjectType;
type ParentOptions = ReadingBlockOptions & RichTextOptions;
type VoicingRichTextOptions = SelfOptions & ParentOptions;

class VoicingRichText extends ReadingBlock( RichText, 1 ) {

  public constructor( text: string, options?: VoicingRichTextOptions ) {
    options = optionize<VoicingRichTextOptions, SelfOptions, ParentOptions>()( {

      // {string|null} - if provided, alternative text that will be read that is different from the
      // visually displayed text
      readingBlockNameResponse: null,

      // pdom
      innerContent: text,

      // voicing
      // default tag name for a ReadingBlock, but there are cases where you may want to override this (such as
      // RichText links)
      readingBlockTagName: 'button'
    }, options );

    // Options that use other options
    options = merge( options, {
      readingBlockNameResponse: options.readingBlockNameResponse || text
    } );

    super( text );

    this.focusHighlight = new ReadingBlockHighlight( this );

    this.mutate( options );
  }
}

scenery.register( 'VoicingRichText', VoicingRichText );
export default VoicingRichText;
