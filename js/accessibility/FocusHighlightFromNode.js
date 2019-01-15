// Copyright 2017, University of Colorado Boulder

/**
 * A FocusHighlightPath subtype that is based around a Node. The focusHighlight is constructed based on the bounds of
 * the node. Handles transformations so that when the source node is transformed, the FocusHighlightFromNode will
 * updated be as well.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var FocusHighlightPath = require( 'SCENERY/accessibility/FocusHighlightPath' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Shape = require( 'KITE/Shape' );

  /**
   *
   * @param {Node|null} node
   * @param {Object} [options]
   * @extends FocusHighlightPath
   * @constructor
   */
  function FocusHighlightFromNode( node, options ) {

    options = _.extend( {

      // {boolean} - if true, highlight will surround local bounds
      useLocalBounds: true,

      // line width options, one for each highlight, will be calculated based on transform of this path unless provided
      outerLineWidth: null,
      innerLineWidth: null,

      // {null|number] - default value is function of node transform (minus translation), but can be set explicitly
      // see FocusHighlightPath.getDilationCoefficient()
      dilationCoefficient: null,

      // {boolean} - if true, dilation for bounds around node will increase, see setShapeFromNode()
      useGroupDilation: false
    }, options );

    this.useLocalBounds = options.useLocalBounds; // @private
    this.useGroupDilation = options.useGroupDilation; // @private
    this.dilationCoefficient = options.dilationCoefficient; // @private

    // @private {Node|null}
    this.sourceNode = node;

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

      // Figure out how much dilation to apply to the focus highlight around the node, calculated unless specified
      // with options
      var defaultDilationCoefficient = ( this.useGroupDilation ? FocusHighlightPath.getGroupDilationCoefficient( node ) :
                                         FocusHighlightPath.getDilationCoefficient( node ) );
      var dilationCoefficient = this.dilationCoefficient || defaultDilationCoefficient;
      var dilatedBounds = this.nodeBounds.dilated( dilationCoefficient );

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
    },

    /**
     * Return the trail to the source node being used for this focus highlight. Assists in observing transforms applied
     * to the source node so that the FocusHighlightFromNode can update accordingly.
     *
     * @public (scenery-internal)
     * @returns {Trail}
     */
    getUniqueHighlightTrail: function() {
      assert && assert( this.sourceNode.instances.length <= 1, 'sourceNode cannot use DAG, must have single trail.' );
      return this.sourceNode.getUniqueTrail();
    }
  } );
} );