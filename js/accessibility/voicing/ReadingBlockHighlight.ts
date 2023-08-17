// Copyright 2021-2023, University of Colorado Boulder

/**
 * A focus highlight for the voicing prototype. Has a unique color to indicate that focus is not around something
 * that is interactive, but can be read with activation.
 *
 * This should generally be used for otherwise NON interactive things that have voicing. Normally focusable things
 * should have the default focus highlight.
 * @author Jesse Greenberg
 */

import optionize, { EmptySelfOptions } from '../../../../phet-core/js/optionize.js';
import { HighlightFromNode, HighlightFromNodeOptions, Node, scenery } from '../../imports.js';

type SelfOptions = EmptySelfOptions;
type ReadingBlockHighlightOptions = SelfOptions & HighlightFromNodeOptions;

class ReadingBlockHighlight extends HighlightFromNode {
  public constructor( node: Node, providedOptions?: ReadingBlockHighlightOptions ) {

    const options = optionize<ReadingBlockHighlightOptions, SelfOptions, HighlightFromNodeOptions>()( {
      outerStroke: 'grey',
      innerStroke: 'black'
    }, providedOptions );

    super( node, options );
  }
}

scenery.register( 'ReadingBlockHighlight', ReadingBlockHighlight );
export default ReadingBlockHighlight;