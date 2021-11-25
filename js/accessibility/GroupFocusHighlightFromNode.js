// Copyright 2018-2020, University of Colorado Boulder

/**
 * A FocusHighlightPath subtype that is based around a Node, with styling that makes it look like a group focus
 * highlight.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import merge from '../../../phet-core/js/merge.js';
import { scenery, FocusHighlightFromNode, FocusHighlightPath } from '../imports.js';

class GroupFocusHighlightFromNode extends FocusHighlightFromNode {
  /**
   * @param {Node|null} node
   * @param {Object} [options]
   */
  constructor( node, options ) {

    options = merge( {
      outerStroke: FocusHighlightPath.OUTER_LIGHT_GROUP_FOCUS_COLOR,
      innerStroke: FocusHighlightPath.INNER_LIGHT_GROUP_FOCUS_COLOR,

      outerLineWidth: FocusHighlightPath.GROUP_OUTER_LINE_WIDTH,
      innerLineWidth: FocusHighlightPath.GROUP_INNER_LINE_WIDTH,

      useGroupDilation: true
    }, options );

    super( node, options );
  }
}

scenery.register( 'GroupFocusHighlightFromNode', GroupFocusHighlightFromNode );
export default GroupFocusHighlightFromNode;