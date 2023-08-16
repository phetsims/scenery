// Copyright 2021-2022, University of Colorado Boulder

/**
 * The highlight with styling used for ReadingBlocks when the are "activated" and
 * the Voicing framework is speaking the content for a Node that composes ReadingBlock.
 *
 * @author Jesse Greenberg
 */

import optionize, { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import { HighlightFromNode, HighlightFromNodeOptions, Node, scenery } from '../../imports.js';

// constants
const ACTIVATED_HIGHLIGHT_COLOR = 'rgba(255,255,0,0.5)';


type SelfOptions = EmptySelfOptions;
type ActivatedReadingBlockHighlightOptions = SelfOptions & HighlightFromNodeOptions;

class ActivatedReadingBlockHighlight extends HighlightFromNode {
  public static readonly ACTIVATED_HIGHLIGHT_COLOR = ACTIVATED_HIGHLIGHT_COLOR;

  public constructor( node: Node | null, providedOptions?: ActivatedReadingBlockHighlightOptions ) {

    const options = optionize<ActivatedReadingBlockHighlightOptions, SelfOptions, HighlightFromNodeOptions>()( {
      innerStroke: null,
      outerStroke: null,
      fill: ACTIVATED_HIGHLIGHT_COLOR
    }, providedOptions );

    super( node, options );
  }
}

scenery.register( 'ActivatedReadingBlockHighlight', ActivatedReadingBlockHighlight );
export default ActivatedReadingBlockHighlight;