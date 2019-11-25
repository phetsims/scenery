// Copyright 2017-2019, University of Colorado Boulder

/**
 * A FocusHighlightPath subtype that is based around a Node. The focusHighlight is constructed based on the bounds of
 * the node. Handles transformations so that when the source node is transformed, the FocusHighlightFromNode will
 * updated be as well.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  const FocusHighlightPath = require( 'SCENERY/accessibility/FocusHighlightPath' );
  const inherit = require( 'PHET_CORE/inherit' );
  const merge = require( 'PHET_CORE/merge' );
  const scenery = require( 'SCENERY/scenery' );
  const Shape = require( 'KITE/Shape' );

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

    if ( node ) {
      this.setShapeFromNode( node );
    }
  }

  scenery.register( 'FocusHighlightFromNode', FocusHighlightFromNode );

  return inherit( FocusHighlightPath, FocusHighlightFromNode, {

    /**
     * Update the focusHighlight shape on the path given the node passed in. Depending on options supplied to this
     * FocusHighlightFromNode, the shape will surround the node's bounds or its local bounds, dilated by an amount
     * that is dependent on whether or not this highlight is for group content or for the node itself. See
     * Accessibility.setGroupFocusHighlight() for more information on group highlights.
     *
     * @param {Node} node
     */
    setShapeFromNode: function( node ) {
      this.nodeBounds = this.useLocalBounds ? node.localBounds : node.bounds;

      let dilationCoefficient = this.dilationCoefficient;

      // Figure out how much dilation to apply to the focus highlight around the node, calculated unless specified
      // with options
      if ( this.dilationCoefficient === null ) {
        dilationCoefficient = ( this.useGroupDilation ? FocusHighlightPath.getGroupDilationCoefficient( node ) :
                              FocusHighlightPath.getDilationCoefficient( node ) );
      }
      const dilatedBounds = this.nodeBounds.dilated( dilationCoefficient );

      // Update the line width of the focus highlight based on the transform of the node
      this.updateLineWidthFromNode( node );
      this.setShape( Shape.bounds( dilatedBounds ) );
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
} );