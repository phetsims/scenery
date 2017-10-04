// Copyright 2017, University of Colorado Boulder

/**
 * A Node for a focus highlight that takes a shape and creates a Path with the default styling of a focus highlight
 * for a11y. The FocusHighlight has two paths.  The FocusHighlight path is an 'outer' highlight that is a little
 * lighter in color and transparency.  It as a child 'inner' path that is darker and more opaque, which gives the
 * focus highlight the illusion that it fades out.
 *
 * @author - Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

define( function( require ) {
  'use strict';

  // modules
  var Color = require( 'SCENERY/util/Color' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Path = require( 'SCENERY/nodes/Path' );
  var scenery = require( 'SCENERY/scenery' );
  var Vector2 = require( 'DOT/Vector2' );

  // constants
  var FOCUS_COLOR = new Color( 'rgba(212,19,106,0.5)' );
  var INNER_FOCUS_COLOR = new Color( 'rgba(250,40,135,0.9)' );

  // Determined by inspection, base widths of focus highlight, transform of shape/bounds will change highlight line width
  var INNER_LINE_WIDTH_BASE = 2.5;
  var OUTER_LINE_WIDTH_BASE = 4;

  /**
   * @constructor
   *
   * @param {Shape} shape - the shape for the focus highlight
   * @param {Object} options
   */
  function FocusHighlightPath( shape, options ) {

    Path.call( this, shape );

    options = _.extend( {
      outerStroke: FOCUS_COLOR,
      innerStroke: INNER_FOCUS_COLOR,

      outerLineWidth: null,
      innerLineWidth: null
    }, options );

    this.options = options; // @private

    // options for this Path, the outer focus highlight
    var outerHighlightOptions = _.extend( {
      stroke: options.outerStroke
    }, options );
    this.mutate( outerHighlightOptions );

    // create the 'inner' focus highlight, the slightly darker and more opaque path that is on the inside
    // of the outer path to give the focus highlight a 'fade-out' appearance
    this.innerHighlightPath = new Path( shape, {
      stroke: options.innerStroke
    } );
    this.addChild( this.innerHighlightPath );

    this.updateLineWidth();
  }

  scenery.register( 'FocusHighlightPath', FocusHighlightPath );

  return inherit( Path, FocusHighlightPath, {

    /**
     * Update the shape of the child path (inner highlight) and this path (outer highlight).
     *
     * @override
     * @param {Shape} shape
     */
    setShape: function( shape ) {
      Path.prototype.setShape.call( this, shape );
      this.innerHighlightPath && this.innerHighlightPath.setShape( shape );
    },

    /**
     * Get a scalar based on the node's transform excluding position.
     * @private
     * @param {Node} [node] - optional node to be used instead of "this"
     * @returns {number}
     */
    getWidthMagnitudeFromTransform: function( node ) {
      node = node || this;
      return node.transform.transformDelta2( Vector2.X_UNIT ).magnitude();
    },

    /**
     * @public
     * Update the line width of both Paths based on transform. Can be overwritten (ridden?) by the options
     * passed in the constructor.
     */
    updateLineWidth: function() {
      this.lineWidth = this.getOuterLineWidth();
      this.innerHighlightPath.lineWidth = this.getInnerLineWidth();
    },

    /**
     * Given a node, return the lineWidth of the focusHighlight that would be supplied.
     * @param {Node} [node] - optional node to base the line width off of, otherwise use "this"
     * @returns {number}
     */
    getOuterLineWidth: function( node ) {
      if ( this.options.outerLineWidth ) {
        return this.options.outerLineWidth;
      }
      node = node || this;
      return OUTER_LINE_WIDTH_BASE / this.getWidthMagnitudeFromTransform( node );
    },

    /**
     * Given a node, return the lineWidth of the inner focusHighlight that would be supplied.
     * @param {Node} [node] - optional node to base the line width off of, otherwise use "this"
     * @returns {number}
     */
    getInnerLineWidth: function( node ) {
      if ( this.options.innerLineWidth ) {
        return this.options.innerLineWidth;
      }
      node = node || this;
      return INNER_LINE_WIDTH_BASE / this.getWidthMagnitudeFromTransform( node );
    }


  } );
} );
