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
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import optionize, { combineOptions } from '../../../phet-core/js/optionize.js';
import { Color, TPaint, Node, Path, PathOptions, scenery, Trail } from '../imports.js';

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

type SelfOptions = {

  // strokes for each highlight
  outerStroke?: TPaint;
  innerStroke?: TPaint;

  // lineWidth for each highlight. If null, the lineWidth will be calculated from the transform of
  // the Node of this highlight (or the transformSourceNode).
  outerLineWidth?: number | null;
  innerLineWidth?: number | null;

  // If specified, this FocusHighlightPath will reposition with transform changes along the unique trail to this source
  // Node. Otherwise you will have to position this highlight node yourself.
  transformSourceNode?: Node | null;
};

// The stroke and linewidth of this path are set with outerLineWidth and outerStroke.
export type FocusHighlightPathOptions = SelfOptions & StrictOmit<PathOptions, 'stroke' | 'lineWidth'>;

class FocusHighlightPath extends Path {

  // The highlight is composed of an "inner" and "outer" path to look nice. These hold each color.
  private _innerHighlightColor: TPaint;
  private _outerHighlightColor: TPaint;

  // Emits whenever this highlight changes.
  public highlightChangedEmitter = new Emitter();

  // See option for documentation.
  public transformSourceNode: Node | null;
  private readonly outerLineWidth: number | null;
  private readonly innerLineWidth: number | null;

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

  /**
   * @param [shape] - the shape for the focus highlight
   * @param [providedOptions]
   */
  public constructor( shape: Shape | string | null, providedOptions?: FocusHighlightPathOptions ) {

    const options = optionize<FocusHighlightPathOptions, SelfOptions, PathOptions>()( {
      outerStroke: OUTER_FOCUS_COLOR,
      innerStroke: INNER_FOCUS_COLOR,
      outerLineWidth: null,
      innerLineWidth: null,
      transformSourceNode: null
    }, providedOptions );

    super( shape );

    this._innerHighlightColor = options.innerStroke;
    this._outerHighlightColor = options.outerStroke;

    const pathOptions = _.pick( options, Object.keys( Path.DEFAULT_PATH_OPTIONS ) ) as PathOptions;

    // Path cannot take null for lineWidth.
    this.innerLineWidth = options.innerLineWidth;
    this.outerLineWidth = options.outerLineWidth;

    this.transformSourceNode = options.transformSourceNode;

    // Assign the 'outer' specific options, and mutate the whole path for pr
    options.stroke = options.outerStroke;
    this.mutate( options );

    const innerHighlightOptions = combineOptions<PathOptions>( {}, pathOptions, {
      stroke: options.innerStroke
    } );

    this.innerHighlightPath = new Path( shape, innerHighlightOptions );
    this.addChild( this.innerHighlightPath );

    this.updateLineWidth();
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
  public makeDashed(): void {
    this.mutateWithInnerHighlight( {
      lineDash: [ 7, 7 ]
    } );
  }

  /**
   * Update the shape of the child path (inner highlight) and this path (outer highlight). Note for the purposes
   * of chaining the outer Path (this) is returned, not the inner Path.
   */
  public override setShape( shape: Shape | string | null ): this {
    super.setShape( shape );
    this.innerHighlightPath && this.innerHighlightPath.setShape( shape );
    this.highlightChangedEmitter && this.highlightChangedEmitter.emit();

    return this;
  }

  /**
   * Update the line width of both Paths based on transform of this Path, or another Node passed in (usually the
   * node that is being highlighted). Can be overridden by the options
   * passed in the constructor.
   *
   * @param [node] - if provided, adjust the line width based on the transform of the node argument
   */
  public updateLineWidth( node?: Node ): void {
    node = node || this; // update based on node passed in or on self.
    this.lineWidth = this.getOuterLineWidth( node );
    this.innerHighlightPath.lineWidth = this.getInnerLineWidth( node );
    this.highlightChangedEmitter.emit();
  }

  /**
   * Given a node, return the lineWidth of this focus highlight.
   */
  public getOuterLineWidth( node: Node ): number {
    if ( this.outerLineWidth ) {
      return this.outerLineWidth;
    }
    return FocusHighlightPath.getOuterLineWidthFromNode( node );
  }

  /**
   * Given a node, return the lineWidth of this focus highlight.
   */
  public getInnerLineWidth( node: Node ): number {
    if ( this.innerLineWidth ) {
      return this.innerLineWidth;
    }
    return FocusHighlightPath.getInnerLineWidthFromNode( node );
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
   * Get the color of the outer highlight for this FocusHighlightPath
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
   */
  public static getInnerLineWidthFromNode( node: Node ): number {
    return INNER_LINE_WIDTH_BASE / FocusHighlightPath.getWidthMagnitudeFromTransform( node );
  }

  /**
   * Get the outer line width of a node, based on its scale and rotation transformation.
   */
  public static getOuterLineWidthFromNode( node: Node ): number {
    return OUTER_LINE_WIDTH_BASE / FocusHighlightPath.getWidthMagnitudeFromTransform( node );
  }

  /**
   * Get a scalar width based on the node's transform excluding position.
   */
  private static getWidthMagnitudeFromTransform( node: Node ): number {
    return node.transform.transformDelta2( Vector2.X_UNIT ).magnitude;
  }

  /**
   * Get the coefficient needed to scale the highlights bounds to surround the node being highlighted elegantly.
   * The highlight is based on a Node's bounds, so it should be scaled out a certain amount so that there is white
   * space between the edge of the component and the beginning (inside edge) of the focusHighlight
   */
  public static getDilationCoefficient( node: Node ): number {
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
   */
  public static getGroupDilationCoefficient( node: Node ): number {
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

scenery.register( 'FocusHighlightPath', FocusHighlightPath );

export default FocusHighlightPath;