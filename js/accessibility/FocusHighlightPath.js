// Copyright 2017-2022, University of Colorado Boulder

/**
 * A Node for a focus highlight that takes a shape and creates a Path with the default styling of a focus highlight
 * for a11y. The FocusHighlight has two paths.  The FocusHighlight path is an 'outer' highlight that is a little
 * lighter in color and transparency.  It as a child 'inner' path that is darker and more opaque, which gives the
 * focus highlight the illusion that it fades out.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Emitter from '../../../axon/js/Emitter.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { DEFAULT_OPTIONS as LINE_STYLES_DEFAULT_OPTIONS } from '../../../kite/js/util/LineStyles.js';
import merge from '../../../phet-core/js/merge.js';
import { Color, Node, Path, scenery } from '../imports.js';

// constants
// default inner and outer strokes for the focus highlight
const OUTER_FOCUS_COLOR = new Color( 'rgba(212,19,106,0.5)' );
const INNER_FOCUS_COLOR = new Color( 'rgba(250,40,135,0.9)' );

// default inner and outer strokes for the group focus highlight, typically over Displays with lighter backgrounds
const INNER_LIGHT_GROUP_FOCUS_COLOR = new Color( 'rgba(233,113,166,1.0)' );
const OUTER_LIGHT_GROUP_FOCUS_COLOR = new Color( 'rgba(233,113,166,1.0)' );

// default inner and outer strokes for the group  focus highlight, typically over Displays with darker backgrounds
const INNER_DARK_GROUP_FOCUS_COLOR = new Color( 'rgba(159,15,80,1.0)' );
const OUTER_DARK_GROUP_FOCUS_COLOR = new Color( 'rgba(159,15,80,1.0)' );

// Determined by inspection, base widths of focus highlight, transform of shape/bounds will change highlight line width
const INNER_LINE_WIDTH_BASE = 2.5;
const OUTER_LINE_WIDTH_BASE = 4;

// determined by inspection, group focus highlights are thinner than default focus highlights
const GROUP_OUTER_LINE_WIDTH = 2;
const GROUP_INNER_LINE_WIDTH = 2;

class FocusHighlightPath extends Path {
  /**
   * @param {Shape} shape - the shape for the focus highlight
   * @param {Object} [options]
   */
  constructor( shape, options ) {

    options = merge( {

      // stroke options,  one for each highlight
      outerStroke: OUTER_FOCUS_COLOR,
      innerStroke: INNER_FOCUS_COLOR,

      // line width options, one for each highlight, will be calculated based on transform of this path unless provided
      outerLineWidth: null,
      innerLineWidth: null,

      // {Node|null} - If specified, this FocusHighlightPath will reposition with transform changes along the unique
      // trail to this source Node. Otherwise you will have to position this highlight node yourself.
      transformSourceNode: null,

      // TODO: this could use nested options
      // remaining paintable options applied to both highlights
      lineDash: [],
      lineCap: LINE_STYLES_DEFAULT_OPTIONS.lineCap,
      lineJoin: LINE_STYLES_DEFAULT_OPTIONS.lineJoin,
      miterLimit: LINE_STYLES_DEFAULT_OPTIONS.miterLimit,
      lineDashOffset: LINE_STYLES_DEFAULT_OPTIONS.lineDashOffset
    }, options );

    super( shape );

    // @private {PaintDef}
    this._innerHighlightColor = options.innerStroke;
    this._outerHighlightColor = options.outerStroke;

    // @public - emitted whenever this highlight changes
    this.highlightChangedEmitter = new Emitter();

    this.options = options; // @private TODO: only assign individual options to 'this'.

    // @public {Node|null} - see options for documentation
    this.transformSourceNode = options.transformSourceNode;

    // options for this Path, the outer focus highlight
    const outerHighlightOptions = merge( {
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

  /**
   * Mutating convenience function to mutate both the inner highlight also
   * @public
   */
  mutateWithInnerHighlight( options ) {
    super.mutate( options );
    this.innerHighlightPath && this.innerHighlightPath.mutate( options );
    this.highlightChangedEmitter.emit();
  }

  /**
   * mutate the Path to make the stroke dashed, by using `lineDash`
   * @public
   */
  makeDashed() {
    this.mutateWithInnerHighlight( {
      lineDash: [ 7, 7 ]
    } );
  }

  /**
   * Update the shape of the child path (inner highlight) and this path (outer highlight).
   * @public
   *
   * @override
   * @param {Shape} shape
   */
  setShape( shape ) {
    super.setShape( shape );
    this.innerHighlightPath && this.innerHighlightPath.setShape( shape );
    this.highlightChangedEmitter && this.highlightChangedEmitter.emit();
  }

  /**
   * Update the line width of both Paths based on transform of this Path, or another Node passed in (usually the
   * node that is being highlighted). Can be overridden by the options
   * passed in the constructor.
   * @param {Node} [node] - if provided, adjust the line width based on the transform of the node argument
   * @public
   */
  updateLineWidth( node ) {
    node = node || this; // update based on node passed in or on self.
    this.lineWidth = this.getOuterLineWidth( node );
    this.innerHighlightPath.lineWidth = this.getInnerLineWidth( node );
    this.highlightChangedEmitter.emit();
  }

  /**
   * Given a node, return the lineWidth of this focus highlight.
   * @param {Node} node
   * @returns {number}
   * @public
   */
  getOuterLineWidth( node ) {
    if ( this.options.outerLineWidth ) {
      return this.options.outerLineWidth;
    }
    return FocusHighlightPath.getOuterLineWidthFromNode( node );
  }

  /**
   * Given a node, return the lineWidth of this focus highlight.
   * @param {Node} node
   * @returns {number}
   * @public
   */
  getInnerLineWidth( node ) {
    if ( this.options.innerLineWidth ) {
      return this.options.innerLineWidth;
    }
    return FocusHighlightPath.getInnerLineWidthFromNode( node );
  }

  /**
   * Set the inner color of this focus highlight.
   * @public
   * @param {PaintDef} color
   */
  setInnerHighlightColor( color ) {
    this._innerHighlightColor = color;
    this.innerHighlightPath.setStroke( color );
    this.highlightChangedEmitter.emit();
  }

  set innerHighlightColor( color ) { this.setInnerHighlightColor( color ); }

  /**
   * Get the inner color of this focus highlight path.
   * @public
   *
   * @returns {PaintDef}
   */
  getInnerHighlightColor() {
    return this._innerHighlightColor;
  }

  get innerHighlightColor() { return this.getInnerHighlightColor(); }

  /**
   * Set the outer color of this focus highlight.
   * @public
   * @param {PaintDef} color
   */
  setOuterHighlightColor( color ) {
    this._outerHighlightColor = color;
    this.setStroke( color );
    this.highlightChangedEmitter.emit();
  }

  set outerHighlightColor( color ) { this.setOuterHighlightColor( color ); }

  /**
   * Get the color of the outer highlight for this FocusHighlightPath
   * @public
   *
   * @returns {PaintDef}
   */
  getOuterHighlightColor() {
    return this._outerHighlightColor;
  }

  get outerHighlightColor() { return this.getOuterHighlightColor(); }

  /**
   * Return the trail to the transform source node being used for this focus highlight. So that we can observe
   * transforms applied to the source node so that the focus highlight can update accordingly.
   *
   * @public (scenery-internal)
   * @param {Trail} focusedTrail - Trail to focused Node, to help search unique Trail to the transformSourceNode
   * @returns {Trail}
   */
  getUniqueHighlightTrail( focusedTrail ) {
    let uniqueTrail = null;

    // if there is only one instance of transformSourceNode we can just grab its unique Trail
    if ( this.transformSourceNode.instances.length <= 1 ) {
      uniqueTrail = this.transformSourceNode.getUniqueTrail();
    }
    else {

      // there are multiple Trails to the focused Node, try to use the one that goes through both the focused trail
      // and the transformSourceNode (a common case).
      const extendedTrails = this.transformSourceNode.getTrails().filter( trail => trail.isExtensionOf( focusedTrail, true ) );

      // If the trail to the transformSourceNode is not unique, does not go through the focused Node, or has
      // multiple Trails that go through the focused Node it is impossible to determine the Trail to use for the
      // highlight. Either avoid DAG for the transformSourceNode or use a FocusHighlightPath without
      // transformSourceNode.
      assert && assert( extendedTrails.length === 1,
        'No unique trail to highlight, either avoid DAG for transformSourceNode or don\'t use transformSourceNode with FocusHighlightPath'
      );

      uniqueTrail = extendedTrails[ 0 ];
    }

    assert && assert( uniqueTrail, 'no unique Trail found for getUniqueHighlightTrail' );
    return uniqueTrail;
  }


  /**
   * Get the outer line width of a focus highlight based on the node's scale and rotation transform information.
   * @public
   * @static
   *
   * @param {Node} node
   * @returns {number}
   */
  static getInnerLineWidthFromNode( node ) {
    return INNER_LINE_WIDTH_BASE / FocusHighlightPath.getWidthMagnitudeFromTransform( node );
  }

  /**
   * Get the outer line width of a node, based on its scale and rotation transformation.
   * @public
   * @static
   *
   * @param {Node} node
   * @returns {number}
   */
  static getOuterLineWidthFromNode( node ) {
    return OUTER_LINE_WIDTH_BASE / FocusHighlightPath.getWidthMagnitudeFromTransform( node );
  }

  /**
   * Get a scalar width based on the node's transform excluding position.
   * @private
   * @static
   *
   * @param {Node} node
   * @returns {number}
   */
  static getWidthMagnitudeFromTransform( node ) {
    return node.transform.transformDelta2( Vector2.X_UNIT ).magnitude;
  }

  /**
   * Get the coefficient needed to scale the highlights bounds to surround the node being highlighted elegantly.
   * The highlight is based on a Node's bounds, so it should be scaled out a certain amount so that there is white
   * space between the edge of the component and the beginning (inside edge) of the focusHighlight
   * @public
   *
   * @param {Node} node
   * @returns {number}
   */
  static getDilationCoefficient( node ) {
    assert && assert( node instanceof Node );
    const widthOfFocusHighlight = FocusHighlightPath.getOuterLineWidthFromNode( node );

    // Dilating half of the focus highlight width will make the inner edge of the focus highlight at the bounds
    // of the node being highlighted.
    const scalarToEdgeOfBounds = 0.5;

    // Dilate the focus highlight slightly more to give whitespace in between the node being highlighted's bounds and
    // the inner edge of the highlight.
    const whiteSpaceScalar = 0.25;

    return widthOfFocusHighlight * ( scalarToEdgeOfBounds + whiteSpaceScalar );
  }

  /**
   * Get the dilation coefficient for a group focus highlight, which extends even further beyond node bounds
   * than a regular focus highlight. The group focus highlight goes around a node whenever its descendant has focus,
   * so this will always surround the normal focus highlight.
   * @public
   *
   * @param {Node} node
   * @returns {number}
   */
  static getGroupDilationCoefficient( node ) {
    const widthOfFocusHighlight = FocusHighlightPath.getOuterLineWidthFromNode( node );

    // Dilating half of the focus highlight width will make the inner edge of the focus highlight at the bounds
    // of the node being highlighted.
    const scalarToEdgeOfBounds = 0.5;

    // Dilate the group focus highlight slightly more to give whitespace in between the node being highlighted's
    // bounds and the inner edge of the highlight.
    const whiteSpaceScalar = 1.4;

    return widthOfFocusHighlight * ( scalarToEdgeOfBounds + whiteSpaceScalar );
  }
}


// @public
// @static defaults available for custom highlights
FocusHighlightPath.OUTER_FOCUS_COLOR = OUTER_FOCUS_COLOR;
FocusHighlightPath.INNER_FOCUS_COLOR = INNER_FOCUS_COLOR;

FocusHighlightPath.INNER_LIGHT_GROUP_FOCUS_COLOR = INNER_LIGHT_GROUP_FOCUS_COLOR;
FocusHighlightPath.OUTER_LIGHT_GROUP_FOCUS_COLOR = OUTER_LIGHT_GROUP_FOCUS_COLOR;

FocusHighlightPath.INNER_DARK_GROUP_FOCUS_COLOR = INNER_DARK_GROUP_FOCUS_COLOR;
FocusHighlightPath.OUTER_DARK_GROUP_FOCUS_COLOR = OUTER_DARK_GROUP_FOCUS_COLOR;

FocusHighlightPath.GROUP_OUTER_LINE_WIDTH = GROUP_OUTER_LINE_WIDTH;
FocusHighlightPath.GROUP_INNER_LINE_WIDTH = GROUP_INNER_LINE_WIDTH;

scenery.register( 'FocusHighlightPath', FocusHighlightPath );

export default FocusHighlightPath;