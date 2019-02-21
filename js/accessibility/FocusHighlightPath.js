// Copyright 2017, University of Colorado Boulder

/**
 * A Node for a focus highlight that takes a shape and creates a Path with the default styling of a focus highlight
 * for a11y. The FocusHighlight has two paths.  The FocusHighlight path is an 'outer' highlight that is a little
 * lighter in color and transparency.  It as a child 'inner' path that is darker and more opaque, which gives the
 * focus highlight the illusion that it fades out.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

define( function( require ) {
  'use strict';

  // modules
  var Color = require( 'SCENERY/util/Color' );
  var Emitter = require( 'AXON/Emitter' );
  var inherit = require( 'PHET_CORE/inherit' );
  var LineStyles = require( 'KITE/util/LineStyles' );
  var Path = require( 'SCENERY/nodes/Path' );
  var scenery = require( 'SCENERY/scenery' );
  var Vector2 = require( 'DOT/Vector2' );

  // constants
  // default inner and outer strokes for the focus highlight
  var OUTER_FOCUS_COLOR = new Color( 'rgba(212,19,106,0.5)' );
  var INNER_FOCUS_COLOR = new Color( 'rgba(250,40,135,0.9)' );

  // default inner and outer strokes for the group focus highlight, typically over Displays with lighter backgrounds
  var INNER_LIGHT_GROUP_FOCUS_COLOR = new Color( 'rgba(233,113,166,1.0)' );
  var OUTER_LIGHT_GROUP_FOCUS_COLOR = new Color( 'rgba(233,113,166,1.0)' );

  // default inner and outer strokes for the group  focus highlight, typically over Displays with darker backgrounds
  var INNER_DARK_GROUP_FOCUS_COLOR = new Color( 'rgba(159,15,80,1.0)' );
  var OUTER_DARK_GROUP_FOCUS_COLOR = new Color( 'rgba(159,15,80,1.0)' );

  // Determined by inspection, base widths of focus highlight, transform of shape/bounds will change highlight line width
  var INNER_LINE_WIDTH_BASE = 2.5;
  var OUTER_LINE_WIDTH_BASE = 4;

  // determined by inspection, group focus highlights are thinner than default focus highlights
  var GROUP_OUTER_LINE_WIDTH = 2;
  var GROUP_INNER_LINE_WIDTH = 2;

  /**
   * @constructor
   *
   * @param {Shape} shape - the shape for the focus highlight
   * @param {Object} options
   */
  function FocusHighlightPath( shape, options ) {

    options = _.extend( {

      // stroke options,  one for each highlight
      outerStroke: OUTER_FOCUS_COLOR,
      innerStroke: INNER_FOCUS_COLOR,

      // line width options, one for each highlight, will be calculated based on transform of this path unless provided
      outerLineWidth: null,
      innerLineWidth: null,

      // TODO: this could use nested options
      // remaining paintable options applied to both highlights
      lineDash: [],
      lineCap: LineStyles.DEFAULT_OPTIONS.lineCap,
      lineJoin: LineStyles.DEFAULT_OPTIONS.lineJoin,
      miterLimit: LineStyles.DEFAULT_OPTIONS.miterLimit,
      lineDashOffset: LineStyles.DEFAULT_OPTIONS.lineDashOffset
    }, options );

    // @private {PaintDef}
    this._innerHighlightColor = options.innerStroke;
    this._outerHighlightColor = options.outerStroke;

    // @public - emitted whenever this highlight changes
    this.highlightChangedEmitter = new Emitter();

    Path.call( this, shape );

    this.options = options; // @private TODO: only assign individual options to 'this'.

    // options for this Path, the outer focus highlight
    var outerHighlightOptions = _.extend( {
      stroke: options.outerStroke
    }, options ); // TODO: Should this overwrite the "stroke" given in the options rather than extend?
    this.mutate( outerHighlightOptions );

    // create the 'inner' focus highlight, the slightly darker and more opaque path that is on the inside
    // of the outer path to give the focus highlight a 'fade-out' appearance
    this.innerHighlightPath = new Path( shape, {
      stroke: options.innerStroke,
      lineDash: options.lineDash,
      lineCap: options.lineCap,
      lineJoin: options.lineJoin,
      miterLimit: options.miterLimit,
      lineDashOffset: options.lineDashOffset
    } );
    this.addChild( this.innerHighlightPath );

    this.updateLineWidth();
  }

  scenery.register( 'FocusHighlightPath', FocusHighlightPath );

  return inherit( Path, FocusHighlightPath, {

    /**
     * Mutating convenience function to mutate both the inner highlight also
     * @public
     */
    mutateWithInnerHighlight: function( options ) {
      Path.prototype.mutate.call( this, options );
      this.innerHighlightPath && this.innerHighlightPath.mutate( options );
      this.highlightChangedEmitter.emit();
    },

    /**
     * mutate the Path to make the stroke dashed, by using `lineDash`
     * @public
     */
    makeDashed: function() {
      this.mutateWithInnerHighlight( {
        lineDash: [ 7, 7 ]
      } );
    },

    /**
     * Update the shape of the child path (inner highlight) and this path (outer highlight).
     *
     * @override
     * @param {Shape} shape
     */
    setShape: function( shape ) {
      Path.prototype.setShape.call( this, shape );
      this.innerHighlightPath && this.innerHighlightPath.setShape( shape );
      this.highlightChangedEmitter.emit();
    },

    /**
     * Update the line width of both Paths based on transform of this Path, or another Node passed in (usually the
     * node that is being highlighted). Can be overridden by the options
     * passed in the constructor.
     * @param {Node} [node] - if provided, adjust the line width based on the transform of the node argument
     * @public
     */
    updateLineWidth: function( node ) {
      node = node || this; // update based on node passed in or on self.
      this.lineWidth = this.getOuterLineWidth( node );
      this.innerHighlightPath.lineWidth = this.getInnerLineWidth( node );
      this.highlightChangedEmitter.emit();
    },

    /**
     * Given a node, return the lineWidth of this focus highlight.
     * @param {Node} node
     * @returns {number}
     * @public
     */
    getOuterLineWidth: function( node ) {
      if ( this.options.outerLineWidth ) {
        return this.options.outerLineWidth;
      }
      return FocusHighlightPath.getOuterLineWidthFromNode( node );
    },

    /**
     * Given a node, return the lineWidth of this focus highlight.
     * @param {Node} node
     * @returns {number}
     * @public
     */
    getInnerLineWidth: function( node ) {
      if ( this.options.innerLineWidth ) {
        return this.options.innerLineWidth;
      }
      return FocusHighlightPath.getInnerLineWidthFromNode( node );
    },

    /**
     * Set the inner color of this focus highlight.
     * @public
     * @param {PaintDef} color
     */
    setInnerHighlightColor: function( color ) {
      this._innerHighlightColor = color;
      this.innerHighlightPath.setStroke( color );
      this.highlightChangedEmitter.emit();
    },
    set innerHighlightColor( color ) { this.setInnerHighlightColor( color ); },

    /**
     * Get the inner color of this focus highlight path.
     * @public
     *
     * @returns {PaintDef}
     */
    getInnerHighlightColor: function() {
      return this._innerHighlightColor;
    },
    get innerHighlightColor() { return this.getInnerHighlightColor(); },

    /**
     * Set the outer color of this focus highlight.
     * @public
     * @param {PaintDef} color
     */
    setOuterHighlightColor: function( color ) {
      this._outerHighlightColor = color;
      this.setStroke( color );
      this.highlightChangedEmitter.emit();
    },
    set outerHighlightColor( color ) { this.setOuterHighlightColor( color ); },

    /**
     * Get the color of the outer highlight for this FocusHighlightPath
     * @public
     *
     * @returns {PaintDef}
     */
    getOuterHighlightColor: function() {
      return this._outerHighlightColor;
    },
    get outerHighlightColor() { return this.getOuterHighlightColor(); }

  }, {

    // @public
    // @static defaults available for custom highlights
    OUTER_FOCUS_COLOR: OUTER_FOCUS_COLOR,
    INNER_FOCUS_COLOR: INNER_FOCUS_COLOR,

    INNER_LIGHT_GROUP_FOCUS_COLOR: INNER_LIGHT_GROUP_FOCUS_COLOR,
    OUTER_LIGHT_GROUP_FOCUS_COLOR: OUTER_LIGHT_GROUP_FOCUS_COLOR,

    INNER_DARK_GROUP_FOCUS_COLOR: INNER_DARK_GROUP_FOCUS_COLOR,
    OUTER_DARK_GROUP_FOCUS_COLOR: OUTER_DARK_GROUP_FOCUS_COLOR,

    GROUP_OUTER_LINE_WIDTH: GROUP_OUTER_LINE_WIDTH,
    GROUP_INNER_LINE_WIDTH: GROUP_INNER_LINE_WIDTH,

    /**
     * Get the outer line width of a focus highlight based on the node's scale and rotation transform information.
     * @public
     * @static
     *
     * @param {Node} node
     * @returns {number}
     */
    getInnerLineWidthFromNode: function( node ) {
      return INNER_LINE_WIDTH_BASE / FocusHighlightPath.getWidthMagnitudeFromTransform( node );
    },

    /**
     * Get the outer line width of a node, based on its scale and rotation transformation.
     * @public
     * @static
     *
     * @param {Node} node
     * @returns {number}
     */
    getOuterLineWidthFromNode: function( node ) {
      return OUTER_LINE_WIDTH_BASE / FocusHighlightPath.getWidthMagnitudeFromTransform( node );
    },

    /**
     * Get a scalar width based on the node's transform excluding position.
     * @private
     * @static
     *
     * @param {Node} node
     * @returns {number}
     */
    getWidthMagnitudeFromTransform: function( node ) {
      return node.transform.transformDelta2( Vector2.X_UNIT ).magnitude;
    },

    /**
     * Get the coefficient needed to scale the highlights bounds to surround the node being highlighted elegantly.
     * The highlight is based on a Node's bounds, so it should be scaled out a certain amount so that there is white
     * space between the edge of the component and the beginning (inside edge) of the focusHighlight
     * @param node
     * @returns {number}
     */
    getDilationCoefficient: function( node ) {
      var widthOfFocusHighlight = FocusHighlightPath.getOuterLineWidthFromNode( node );

      // Dilating half of the focus highlight width will make the inner edge of the focus highlight at the bounds
      // of the node being highlighted.
      var scalarToEdgeOfBounds = .5;

      // Dilate the focus highlight slightly more to give whitespace in between the node being highlighted's bounds and
      // the inner edge of the highlight.
      var whiteSpaceScalar = .25;

      return widthOfFocusHighlight * ( scalarToEdgeOfBounds + whiteSpaceScalar );
    },

    /**
     * Get the dilation coefficient for a group focus highlight, which extends even further beyond node bounds
     * than a regular focus highlight. The group focus highlight goes around a node whenever its descendant has focus,
     * so this will always surround the normal focus highlight.
     *
     * @param {Node} node
     *
     * @returns {number}
     */
    getGroupDilationCoefficient: function( node ) {
      var widthOfFocusHighlight = FocusHighlightPath.getOuterLineWidthFromNode( node );

      // Dilating half of the focus highlight width will make the inner edge of the focus highlight at the bounds
      // of the node being highlighted.
      var scalarToEdgeOfBounds = .5;

      // Dilate the group focus highlight slightly more to give whitespace in between the node being highlighted's 
      // bounds and the inner edge of the highlight.
      var whiteSpaceScalar = 1.4;

      return widthOfFocusHighlight * ( scalarToEdgeOfBounds + whiteSpaceScalar );
    }
  } );
} );
