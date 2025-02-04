// Copyright 2018-2025, University of Colorado Boulder

/**
 * A HighlightPath subtype that is based around a Node, with styling that makes it look like a group focus
 * highlight.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import optionize, { EmptySelfOptions } from '../../../phet-core/js/optionize.js';
import type { HighlightFromNodeOptions } from '../accessibility/HighlightFromNode.js';
import HighlightFromNode from '../accessibility/HighlightFromNode.js';
import HighlightPath from '../accessibility/HighlightPath.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';

type SelfOptions = EmptySelfOptions;
type GroupHighlightFromNodeOptions = HighlightFromNodeOptions;

class GroupHighlightFromNode extends HighlightFromNode {
  public constructor( node: Node | null, providedOptions?: GroupHighlightFromNodeOptions ) {

    const options = optionize<GroupHighlightFromNodeOptions, SelfOptions, HighlightFromNodeOptions>()( {
      outerStroke: HighlightPath.OUTER_LIGHT_GROUP_FOCUS_COLOR,
      innerStroke: HighlightPath.INNER_LIGHT_GROUP_FOCUS_COLOR,

      outerLineWidth: HighlightPath.GROUP_OUTER_LINE_WIDTH,
      innerLineWidth: HighlightPath.GROUP_INNER_LINE_WIDTH,

      useGroupDilation: true
    }, providedOptions );

    super( node, options );
  }
}

scenery.register( 'GroupHighlightFromNode', GroupHighlightFromNode );
export default GroupHighlightFromNode;