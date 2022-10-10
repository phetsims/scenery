// Copyright 2022, University of Colorado Boulder

/**
 * A convenience superclass for a Node composed with InteractiveHighlighting. Helpful when using this superclass is
 * easier than the trait pattern. And some devs generally prefer traditional inheritance.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import { InteractiveHighlighting, InteractiveHighlightingOptions, Node, NodeOptions, scenery } from '../../../imports.js';

export type InteractiveHighlightingNodeOptions = InteractiveHighlightingOptions & NodeOptions;

class InteractiveHighlightingNode extends InteractiveHighlighting( Node ) {
  public constructor( options?: InteractiveHighlightingNodeOptions ) {
    super( options );
  }
}

scenery.register( 'InteractiveHighlightingNode', InteractiveHighlightingNode );
export default InteractiveHighlightingNode;
