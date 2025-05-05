// Copyright 2017-2025, University of Colorado Boulder

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
import TReadOnlyProperty from '../../../axon/js/TReadOnlyProperty.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import optionize, { combineOptions } from '../../../phet-core/js/optionize.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import animatedPanZoomSingleton from '../listeners/animatedPanZoomSingleton.js';
import Node from '../nodes/Node.js';
import type { InputShape, PathOptions } from '../nodes/Path.js';
import Path from '../nodes/Path.js';
import scenery from '../scenery.js';
import Color from '../util/Color.js';
import type TPaint from '../util/TPaint.js';
import Trail from '../util/Trail.js';

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
const LINE_DASH_BASE = 7;

// determined by inspection, group focus highlights are thinner than default focus highlights
const GROUP_OUTER_LINE_WIDTH = 2;
const GROUP_INNER_LINE_WIDTH = 2;

type SelfOptions = {

  // strokes for each highlight
  outerStroke?: TPaint;
  innerStroke?: TPaint;

  // lineWidth for each highlight. If null, the lineWidth will be calculated from the transform of
  // the Node of this highlight (or the transformSourceNode).
  outerLineWidth?: number | null;
  innerLineWidth?: number | null;

  // A lineDash to apply to the highlight. If specified, this will override the default lineDash for the highlight.
  // If null, the lineDash will be calculated from the transform of the Node of this highlight (or the
  // transformSourceNode). If an override is provided, no transformation will be applied to the lineDash.
  lineDashOverride?: number[] | null;

  // If true, the highlight will appear dashed with a lineDash effect. Used often by PhET to indicate that an
  // interactive component is currently picked up and being manipulated by the user.
  dashed?: boolean;

  // If specified, this HighlightPath will reposition with transform changes along the unique trail to this source
  // Node. Otherwise you will have to position this highlight node yourself.
  transformSourceNode?: Node | null;
};

// The stroke and linewidth of this path are set with outerLineWidth and outerStroke.
export type HighlightPathOptions = SelfOptions & StrictOmit<PathOptions, 'stroke' | 'lineWidth'>;

class HighlightPath extends Path {

  // The highlight is composed of an "inner" and "outer" path to look nice. These hold each color.
  private _innerHighlightColor: TPaint;
  private _outerHighlightColor: TPaint;

  // Emits whenever this highlight changes.
  public readonly highlightChangedEmitter = new Emitter();

  // See option for documentation.
  public readonly transformSourceNode: Node | null;
  private readonly outerLineWidth: number | null;
  private readonly innerLineWidth: number | null;

  // The line dash for the highlight for setDashed. Will be transformed so that scale Nodes have consistent line dashes.
  private transformedLineDash = [ LINE_DASH_BASE, LINE_DASH_BASE ];
  private readonly lineDashOverride: number[] | null;

  // The 'inner' focus highlight, the (by default) slightly darker and more opaque path that is on the inside of the
  // outer path to give the focus highlight a 'fade-out' appearance
  protected readonly innerHighlightPath: Path;

  public static readonly OUTER_FOCUS_COLOR = OUTER_FOCUS_COLOR;
  public static readonly INNER_FOCUS_COLOR = INNER_FOCUS_COLOR;

  public static readonly INNER_LIGHT_GROUP_FOCUS_COLOR = INNER_LIGHT_GROUP_FOCUS_COLOR;
  public static readonly OUTER_LIGHT_GROUP_FOCUS_COLOR = OUTER_LIGHT_GROUP_FOCUS_COLOR;

  public static readonly INNER_DARK_GROUP_FOCUS_COLOR = INNER_DARK_GROUP_FOCUS_COLOR;
  public static readonly OUTER_DARK_GROUP_FOCUS_COLOR = OUTER_DARK_GROUP_FOCUS_COLOR;

  public static readonly GROUP_OUTER_LINE_WIDTH = GROUP_OUTER_LINE_WIDTH;
  public static readonly GROUP_INNER_LINE_WIDTH = GROUP_INNER_LINE_WIDTH;

  // A scalar describing the layout scale of your application. Highlight line widths are corrected
  // by the layout scale so that they have the same sizes relative to the size of the application.
  public static layoutScale = 1;

  /**
   * @param [shape] - the shape for the focus highlight
   * @param [providedOptions]
   */
  public constructor( shape: InputShape | TReadOnlyProperty<InputShape>, providedOptions?: HighlightPathOptions ) {

    const options = optionize<HighlightPathOptions, SelfOptions, PathOptions>()( {
      outerStroke: OUTER_FOCUS_COLOR,
      innerStroke: INNER_FOCUS_COLOR,
      outerLineWidth: null,
      innerLineWidth: null,
      lineDashOverride: null,
      dashed: false,
      transformSourceNode: null
    }, providedOptions );

    super( shape );

    this._innerHighlightColor = options.innerStroke;
    this._outerHighlightColor = options.outerStroke;

    const pathOptions = _.pick( options, Object.keys( Path.DEFAULT_PATH_OPTIONS ) ) as PathOptions;

    // Path cannot take null for lineWidth.
    this.innerLineWidth = options.innerLineWidth;
    this.outerLineWidth = options.outerLineWidth;
    this.lineDashOverride = options.lineDashOverride;

    this.transformSourceNode = options.transformSourceNode;

    // Assign the 'outer' specific options, and mutate the whole path for pr
    options.stroke = options.outerStroke;
    this.mutate( options );

    const innerHighlightOptions = combineOptions<PathOptions>( {}, pathOptions, {
      stroke: options.innerStroke
    } );

    this.innerHighlightPath = new Path( shape, innerHighlightOptions );
    this.addChild( this.innerHighlightPath );

    if ( options.dashed ) {
      this.setDashed( true );
    }
  }

  /**
   * Mutating convenience function to mutate both the innerHighlightPath and outerHighlightPath.
   */
  public mutateWithInnerHighlight( options: PathOptions ): void {
    super.mutate( options );
    this.innerHighlightPath && this.innerHighlightPath.mutate( options );
    this.highlightChangedEmitter.emit();
  }

  /**
   * Mutate both inner and outer Paths to make the stroke dashed by using `lineDash`.
   */
  public setDashed( dashOn: boolean ): void {

    // If there is a lineDashOverride, use that instead of the transformedLineDash. If the dash is off,
    // set the lineDash to an empty array.
    const lineDash = dashOn ? ( this.lineDashOverride ? this.lineDashOverride : this.transformedLineDash ) : [];

    this.mutateWithInnerHighlight( {
      lineDash: lineDash
    } );
  }

  /**
   * Update the shape of the child path (inner highlight) and this path (outer highlight). Note for the purposes
   * of chaining the outer Path (this) is returned, not the inner Path.
   */
  public override setShape( shape: InputShape ): this {
    super.setShape( shape );
    this.innerHighlightPath && this.innerHighlightPath.setShape( shape );
    this.highlightChangedEmitter && this.highlightChangedEmitter.emit();

    return this;
  }

  /**
   * Update the line width and dashes of both Paths based on transform of this Path, or another Node passed
   * in (usually the node that is being highlighted). Can be overridden by the options
   * passed in the constructor.
   */
  public updateLineWidth( matrix: Matrix3 ): void {
    this.lineWidth = this.getOuterLineWidth( matrix );
    this.innerHighlightPath.lineWidth = this.getInnerLineWidth( matrix );
    this.transformedLineDash = this.getTransformedLineDash( matrix );
    this.highlightChangedEmitter.emit();
  }

  /**
   * Given a transformation matrix, return the lineWidth of this focus highlight (unless a custom
   * lineWidth was specified in the options).
   *
   * Note - this takes a matrix3 instead of a Node because that is already computed by the highlight
   * overlay and we can avoid the extra computation of the Node's local-to-global matrix.
   */
  public getOuterLineWidth( matrix: Matrix3 ): number {
    if ( this.outerLineWidth ) {
      return this.outerLineWidth;
    }
    return HighlightPath.getOuterLineWidthFromMatrix( matrix );
  }

  /**
   * Given a transformation matrix, return the lineWidth of this focus highlight (unless a custom
   * lineWidth was specified in the options).
   *
   * Note - this takes a matrix3 instead of a Node because that is already computed by the highlight
   * overlay and we can avoid the extra computation of the Node's local-to-global matrix.
   */
  public getInnerLineWidth( matrix: Matrix3 ): number {
    if ( this.innerLineWidth ) {
      return this.innerLineWidth;
    }
    return HighlightPath.getInnerLineWidthFromMatrix( matrix );
  }

  /**
   * Given a transformation matrix, return the lineDash of this focus highlight to produce a consistent
   * lineDash effect.
   */
  private getTransformedLineDash( matrix: Matrix3 ): number[] {
    return HighlightPath.getLineDashFromMatrix( matrix );
  }

  /**
   * Set the inner color of this focus highlight.
   */
  public setInnerHighlightColor( color: TPaint ): void {
    this._innerHighlightColor = color;
    this.innerHighlightPath.setStroke( color );
    this.highlightChangedEmitter.emit();
  }

  public set innerHighlightColor( color: TPaint ) { this.setInnerHighlightColor( color ); }

  public get innerHighlightColor(): TPaint { return this.getInnerHighlightColor(); }

  /**
   * Get the inner color of this focus highlight path.
   */
  public getInnerHighlightColor(): TPaint {
    return this._innerHighlightColor;
  }

  /**
   * Set the outer color of this focus highlight.
   */
  public setOuterHighlightColor( color: TPaint ): void {
    this._outerHighlightColor = color;
    this.setStroke( color );
    this.highlightChangedEmitter.emit();
  }

  public set outerHighlightColor( color: TPaint ) { this.setOuterHighlightColor( color ); }

  public get outerHighlightColor(): TPaint { return this.getOuterHighlightColor(); }

  /**
   * Get the color of the outer highlight for this HighlightPath
   */
  public getOuterHighlightColor(): TPaint {
    return this._outerHighlightColor;
  }

  /**
   * Return the trail to the transform source node being used for this focus highlight. So that we can observe
   * transforms applied to the source node so that the focus highlight can update accordingly.
   * (scenery-internal)
   *
   * @param focusedTrail - Trail to focused Node, to help search unique Trail to the transformSourceNode
   */
  public getUniqueHighlightTrail( focusedTrail: Trail ): Trail {
    assert && assert( this.transformSourceNode, 'getUniqueHighlightTrail requires a transformSourceNode' );
    const transformSourceNode = this.transformSourceNode!;

    let uniqueTrail = null;

    // if there is only one instance of transformSourceNode we can just grab its unique Trail
    if ( transformSourceNode.instances.length <= 1 ) {
      uniqueTrail = transformSourceNode.getUniqueTrail();
    }
    else {

      // there are multiple Trails to the focused Node, try to use the one that goes through both the focused trail
      // and the transformSourceNode (a common case).
      const extendedTrails = transformSourceNode.getTrails().filter( trail => trail.isExtensionOf( focusedTrail, true ) );

      // If the trail to the transformSourceNode is not unique, does not go through the focused Node, or has
      // multiple Trails that go through the focused Node it is impossible to determine the Trail to use for the
      // highlight. Either avoid DAG for the transformSourceNode or use a HighlightPath without
      // transformSourceNode.
      assert && assert( extendedTrails.length === 1,
        'No unique trail to highlight, either avoid DAG for transformSourceNode or don\'t use transformSourceNode with HighlightPath'
      );

      uniqueTrail = extendedTrails[ 0 ];
    }

    assert && assert( uniqueTrail, 'no unique Trail found for getUniqueHighlightTrail' );
    return uniqueTrail;
  }


  /**
   * Get the inner line width for a transformation matrix (presumably from the Node being highlighted).
   */
  private static getInnerLineWidthFromMatrix( matrix: Matrix3 ): number {
    return INNER_LINE_WIDTH_BASE / HighlightPath.localToGlobalScaleFromMatrix( matrix );
  }

  /**
   * Get the outer line width for a transformation matrix (presumably from the Node being highlighted).
   */
  private static getOuterLineWidthFromMatrix( matrix: Matrix3 ): number {
    return OUTER_LINE_WIDTH_BASE / HighlightPath.localToGlobalScaleFromMatrix( matrix );
  }

  /**
   * Get the line dash for a transformation matrix (presumably from the Node being highlighted).
   * This is used to make sure that the line dash is consistent across different scales.
   */
  private static getLineDashFromMatrix( matrix: Matrix3 ): number[] {
    const scale = HighlightPath.localToGlobalScaleFromMatrix( matrix );
    return [ LINE_DASH_BASE / scale, LINE_DASH_BASE / scale ];
  }

  /**
   * Get a scalar width to use for the focus highlight based on the global transformation matrix
   * (presumably from the Node being highlighted). This helps make sure that the highlight
   * line width remains consistent even when the Node has some scale applied to it.
   */
  private static localToGlobalScaleFromMatrix( matrix: Matrix3 ): number {

    // The scale value in X of the matrix, without the Vector2 instance from getScaleVector.
    // The scale vector is assumed to be isometric, so we only need to consider the x component.
    return Math.sqrt( matrix.m00() * matrix.m00() + matrix.m10() * matrix.m10() );
  }

  /**
   * Get the coefficient needed to scale the highlights bounds to surround the node being highlighted elegantly.
   * The highlight is based on a Node's bounds, so it should be scaled out a certain amount so that there is white
   * space between the edge of the component and the beginning (inside edge) of the focusHighlight
   */
  public static getDilationCoefficient( matrix: Matrix3 ): number {
    const widthOfFocusHighlight = HighlightPath.getOuterLineWidthFromMatrix( matrix );

    // Dilating half of the focus highlight width will make the inner edge of the focus highlight at the bounds
    // of the node being highlighted.
    const scalarToEdgeOfBounds = 0.5;

    // Dilate the focus highlight slightly more to give whitespace in between the node being highlighted's bounds and
    // the inner edge of the highlight.
    const whiteSpaceScalar = 0.25;

    return widthOfFocusHighlight * ( scalarToEdgeOfBounds + whiteSpaceScalar );
  }

  /**
   * Returns the highlight dilation coefficient when there is no transformation.
   */
  public static getDefaultDilationCoefficient(): number {
    return HighlightPath.getDilationCoefficient( Matrix3.IDENTITY );
  }

  /**
   * Returns the highlight dilation coefficient for a group focus highlight, which is a bit
   * larger than the typical dilation coefficient.
   */
  public static getDefaultGroupDilationCoefficient(): number {
    return HighlightPath.getGroupDilationCoefficient( Matrix3.IDENTITY );
  }

  /**
   * The default highlight line width. The outer line width is wider and can be used as a value for layout. This is the
   * value of the line width without any transformation. The actual value in the global coordinate frame may change
   * based on the pan/zoom of the screen.
   */
  public static getDefaultHighlightLineWidth(): number {
    return OUTER_LINE_WIDTH_BASE;
  }

  /**
   * Get the dilation coefficient for a group focus highlight, which extends even further beyond node bounds
   * than a regular focus highlight. The group focus highlight goes around a node whenever its descendant has focus,
   * so this will always surround the normal focus highlight.
   */
  public static getGroupDilationCoefficient( matrix: Matrix3 ): number {
    const widthOfFocusHighlight = HighlightPath.getOuterLineWidthFromMatrix( matrix );

    // Dilating half of the focus highlight width will make the inner edge of the focus highlight at the bounds
    // of the node being highlighted.
    const scalarToEdgeOfBounds = 0.5;

    // Dilate the group focus highlight slightly more to give whitespace in between the node being highlighted's
    // bounds and the inner edge of the highlight.
    const whiteSpaceScalar = 1.4;

    return widthOfFocusHighlight * ( scalarToEdgeOfBounds + whiteSpaceScalar );
  }

  /**
   * Returns a matrix representing the inverse of the pan/zoom transform, so that the highlight can be drawn in the
   * global coordinate frame. Do not modify this matrix.
   */
  private static getPanZoomCorrectingMatrix(): Matrix3 {
    if ( animatedPanZoomSingleton.initialized ) {
      return animatedPanZoomSingleton.listener.matrixProperty.value.inverted();
    }
    else {
      return Matrix3.IDENTITY;
    }
  }

  /**
   * Returns a matrix that corrects for the layout scale of the application, so that the highlight can be drawn in the
   * global coordinate frame. Do not modify this matrix.
   */
  private static getLayoutCorrectingMatrix(): Matrix3 {
    return Matrix3.scaling( 1 / HighlightPath.layoutScale, 1 / HighlightPath.layoutScale );
  }

  /**
   * Returns a final matrix to use to scale a highlight so that it is in a consistent size relative to the
   * application layout bounds.
   */
  public static getCorrectiveScalingMatrix(): Matrix3 {
    return HighlightPath.getPanZoomCorrectingMatrix().timesMatrix( HighlightPath.getLayoutCorrectingMatrix() );
  }
}

scenery.register( 'HighlightPath', HighlightPath );

export default HighlightPath;