// Copyright 2018-2019, University of Colorado Boulder

/**
 * A FocusHighlightPath subtype that is based around a Node, with styling that makes it look like a group focus
 * highlight.
 * 
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  const FocusHighlightFromNode = require( 'SCENERY/accessibility/FocusHighlightFromNode' );
  const FocusHighlightPath = require( 'SCENERY/accessibility/FocusHighlightPath' );
  const inherit = require( 'PHET_CORE/inherit' );
  const merge = require( 'PHET_CORE/merge' );
  const scenery = require( 'SCENERY/scenery' );

  /**
   *
   * @param {Node|null} node
   * @param {Object} [options]
   * @constructor
   */
  function GroupFocusHighlightFromNode( node, options ) {

    options = merge( {
      outerStroke: FocusHighlightPath.OUTER_LIGHT_GROUP_FOCUS_COLOR,
      innerStroke: FocusHighlightPath.INNER_LIGHT_GROUP_FOCUS_COLOR,

      outerLineWidth: FocusHighlightPath.GROUP_OUTER_LINE_WIDTH,
      innerLineWidth: FocusHighlightPath.GROUP_INNER_LINE_WIDTH,

      useGroupDilation: true
    }, options );

    FocusHighlightFromNode.call( this, node, options );
  }

  scenery.register( 'GroupFocusHighlightFromNode', GroupFocusHighlightFromNode );

  return inherit( FocusHighlightFromNode, GroupFocusHighlightFromNode );
} );