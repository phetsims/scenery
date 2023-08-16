// Copyright 2018-2022, University of Colorado Boulder

/**
 * A HighlightPath subtype that is based around a Node, with styling that makes it look like a group focus
 * highlight.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import optionize, { EmptySelfOptions } from '../../../phet-core/js/optionize.js';
import { FocusHighlightFromNode, FocusHighlightFromNodeOptions, HighlightPath, Node, scenery } from '../imports.js';

type SelfOptions = EmptySelfOptions;
type GroupFocusHighlightFromNodeOptions = FocusHighlightFromNodeOptions;

class GroupFocusHighlightFromNode extends FocusHighlightFromNode {
  public constructor( node: Node | null, providedOptions?: GroupFocusHighlightFromNodeOptions ) {

    const options = optionize<GroupFocusHighlightFromNodeOptions, SelfOptions, FocusHighlightFromNodeOptions>()( {
      outerStroke: HighlightPath.OUTER_LIGHT_GROUP_FOCUS_COLOR,
      innerStroke: HighlightPath.INNER_LIGHT_GROUP_FOCUS_COLOR,

      outerLineWidth: HighlightPath.GROUP_OUTER_LINE_WIDTH,
      innerLineWidth: HighlightPath.GROUP_INNER_LINE_WIDTH,

      useGroupDilation: true
    }, providedOptions );

    super( node, options );
  }
}

scenery.register( 'GroupFocusHighlightFromNode', GroupFocusHighlightFromNode );
export default GroupFocusHighlightFromNode;