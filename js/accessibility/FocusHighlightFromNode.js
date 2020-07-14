// Copyright 2017-2020, University of Colorado Boulder

/**
 * A FocusHighlightPath subtype that is based around a Node. The focusHighlight is constructed based on the bounds of
 * the node. The focusHighlight will update as the Node's bounds changes. Handles transformations so that when the
 * source node is transformed, the FocusHighlightFromNode will
 * updated be as well.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Shape from '../../../kite/js/Shape.js';
import inherit from '../../../phet-core/js/inherit.js';
import merge from '../../../phet-core/js/merge.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import FocusHighlightPath from './FocusHighlightPath.js';

/**
 *
 * @param {Node|null} node
 * @param {Object} [options]
 * @extends FocusHighlightPath
 * @constructor
 */
function FocusHighlightFromNode( node, options ) {

  options = merge( {

    // {boolean} - if true, highlight will surround local bounds
    useLocalBounds: true,

    // {Node|null} - see FocusHighlightPath for more documentation
    transformSourceNode: node,

    // line width options, one for each highlight, will be calculated based on transform of this path unless provided
    outerLineWidth: null,
    innerLineWidth: null,

    // {null|number] - default value is function of node transform (minus translation), but can be set explicitly
    // see FocusHighlightPath.getDilationCoefficient(). A number here refers to the amount in pixels to dilate the
    // focus highlight by
    dilationCoefficient: null,

    // {boolean} - if true, dilation for bounds around node will increase, see setShapeFromNode()
    useGroupDilation: false
  }, options );

  this.useLocalBounds = options.useLocalBounds; // @private
  this.useGroupDilation = options.useGroupDilation; // @private
  this.dilationCoefficient = options.dilationCoefficient; // @private

  FocusHighlightPath.call( this, null, options );

  // @private - from options, will override line width calculations based on the node's size
  this.outerLineWidth = options.outerLineWidth;
  this.innerLineWidth = options.innerLineWidth;

  // @private {Property.<Bounds2>} - keep track to remove listener
  this.observedBoundsProperty = null;

  // @private {function} - keep track of the listener so that it can be removed
  this.boundsListener = null;

  if ( node ) {
    this.setShapeFromNode( node );
  }
}

scenery.register( 'FocusHighlightFromNode', FocusHighlightFromNode );

inherit( FocusHighlightPath, FocusHighlightFromNode, {

  /**
   * Update the focusHighlight shape on the path given the node passed in. Depending on options supplied to this
   * FocusHighlightFromNode, the shape will surround the node's bounds or its local bounds, dilated by an amount
   * that is dependent on whether or not this highlight is for group content or for the node itself. See
   * ParallelDOM.setGroupFocusHighlight() for more information on group highlights.
   *
   * @param {Node} node
   */
  setShapeFromNode: function( node ) {
    assert && assert( node instanceof Node );

    // cleanup the previous listener
    if ( this.observedBoundsProperty ) {
      assert && assert( this.boundsListener, 'should be a listener if there is a previous focusHighlightNode' );
      this.observedBoundsProperty.unlink( this.boundsListener );
    }

    this.observedBoundsProperty = this.useLocalBounds ? node.localBoundsProperty : node.boundsProperty;

    this.boundsListener = bounds => {

      // Ignore setting the shape if we don't yet have finite bounds.
      if ( !bounds.isFinite() ) {
        return;
      }

      let dilationCoefficient = this.dilationCoefficient;

      // Figure out how much dilation to apply to the focus highlight around the node, calculated unless specified
      // with options
      if ( this.dilationCoefficient === null ) {
        dilationCoefficient = ( this.useGroupDilation ? FocusHighlightPath.getGroupDilationCoefficient( node ) :
                                FocusHighlightPath.getDilationCoefficient( node ) );
      }
      const dilatedBounds = bounds.dilated( dilationCoefficient );

      // Update the line width of the focus highlight based on the transform of the node
      this.updateLineWidthFromNode( node );
      this.setShape( Shape.bounds( dilatedBounds ) );
    };
    this.observedBoundsProperty.link( this.boundsListener );
  },

  /**
   * @private
   * Update the line width of both Paths based on transform.
   * @param node
   */
  updateLineWidthFromNode: function( node ) {

    // Default options can override
    this.lineWidth = this.outerLineWidth || FocusHighlightPath.getOuterLineWidthFromNode( node );
    this.innerHighlightPath.lineWidth = this.innerLineWidth || FocusHighlightPath.getInnerLineWidthFromNode( node );
  }
} );

export default FocusHighlightFromNode;