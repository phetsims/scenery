// Copyright 2021-2022, University of Colorado Boulder

/**
 * A vertical/horizontal flow-based layout container.
 *
 * See https://phetsims.github.io/scenery/doc/layout#FlowBox for details
 *
 * FlowBox-only options:
 *   - resize (see https://phetsims.github.io/scenery/doc/layout#FlowBox-resize)
 *   - orientation (see https://phetsims.github.io/scenery/doc/layout#FlowBox-orientation)
 *   - spacing (see https://phetsims.github.io/scenery/doc/layout#FlowBox-spacing)
 *   - lineSpacing (see https://phetsims.github.io/scenery/doc/layout#FlowBox-lineSpacing)
 *   - justify (see https://phetsims.github.io/scenery/doc/layout#FlowBox-justify)
 *   - justifyLines (see https://phetsims.github.io/scenery/doc/layout#FlowBox-justifyLines)
 *   - wrap (see https://phetsims.github.io/scenery/doc/layout#FlowBox-wrap)
 *   - layoutOrigin (see https://phetsims.github.io/scenery/doc/layout#layoutOrigin)
 *
 * FlowBox and layoutOptions options (can be set either in the FlowBox itself, or within its child nodes' layoutOptions):
 *   - align (see https://phetsims.github.io/scenery/doc/layout#FlowBox-align)
 *   - stretch (see https://phetsims.github.io/scenery/doc/layout#FlowBox-stretch)
 *   - grow (see https://phetsims.github.io/scenery/doc/layout#FlowBox-grow)
 *   - margin (see https://phetsims.github.io/scenery/doc/layout#FlowBox-margins)
 *   - xMargin (see https://phetsims.github.io/scenery/doc/layout#FlowBox-margins)
 *   - yMargin (see https://phetsims.github.io/scenery/doc/layout#FlowBox-margins)
 *   - leftMargin (see https://phetsims.github.io/scenery/doc/layout#FlowBox-margins)
 *   - rightMargin (see https://phetsims.github.io/scenery/doc/layout#FlowBox-margins)
 *   - topMargin (see https://phetsims.github.io/scenery/doc/layout#FlowBox-margins)
 *   - bottomMargin (see https://phetsims.github.io/scenery/doc/layout#FlowBox-margins)
 *   - minContentWidth (see https://phetsims.github.io/scenery/doc/layout#FlowBox-minContent)
 *   - minContentHeight (see https://phetsims.github.io/scenery/doc/layout#FlowBox-minContent)
 *   - maxContentWidth (see https://phetsims.github.io/scenery/doc/layout#FlowBox-maxContent)
 *   - maxContentHeight (see https://phetsims.github.io/scenery/doc/layout#FlowBox-maxContent)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import optionize from '../../../../phet-core/js/optionize.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import { FLOW_CONSTRAINT_OPTION_KEYS, FlowCell, FlowConstraint, FlowConstraintOptions, HorizontalLayoutAlign, HorizontalLayoutJustification, LAYOUT_NODE_OPTION_KEYS, LayoutAlign, LayoutNode, LayoutNodeOptions, LayoutOrientation, MarginLayoutCell, Node, REQUIRES_BOUNDS_OPTION_KEYS, scenery, SceneryConstants, SIZABLE_OPTION_KEYS, VerticalLayoutAlign, VerticalLayoutJustification } from '../../imports.js';

// FlowBox-specific options that can be passed in the constructor or mutate() call.
const FLOWBOX_OPTION_KEYS = [
  ...LAYOUT_NODE_OPTION_KEYS,
  ...FLOW_CONSTRAINT_OPTION_KEYS.filter( key => key !== 'excludeInvisible' )
];

const DEFAULT_OPTIONS = {
  orientation: 'horizontal',
  spacing: 0,
  align: 'center',
  stretch: false
} as const;

type ExcludeFlowConstraintOptions = 'excludeInvisible' | 'preferredWidthProperty' | 'preferredHeightProperty' | 'minimumWidthProperty' | 'minimumHeightProperty' | 'layoutOriginProperty';
type SelfOptions = {
  // Controls whether the FlowBox will re-trigger layout automatically after the "first" layout during construction.
  // The FlowBox will layout once after processing the options object, but if resize:false, then after that manual
  // layout calls will need to be done (with updateLayout())
  resize?: boolean;
} & StrictOmit<FlowConstraintOptions, ExcludeFlowConstraintOptions>;

export type FlowBoxOptions = SelfOptions & LayoutNodeOptions;

export default class FlowBox extends LayoutNode<FlowConstraint> {

  // Track the connection between Nodes and cells
  private readonly _cellMap: Map<Node, FlowCell> = new Map<Node, FlowCell>();

  // Listeners that we'll need to remove
  private readonly onChildInserted: ( node: Node, index: number ) => void;
  private readonly onChildRemoved: ( node: Node ) => void;
  private readonly onChildrenReordered: ( minChangeIndex: number, maxChangeIndex: number ) => void;
  private readonly onChildrenChanged: () => void;

  constructor( providedOptions?: FlowBoxOptions ) {
    const options = optionize<FlowBoxOptions, StrictOmit<SelfOptions, Exclude<keyof FlowConstraintOptions, ExcludeFlowConstraintOptions>>, LayoutNodeOptions>()( {
      // Allow dynamic layout by default, see https://github.com/phetsims/joist/issues/608
      excludeInvisibleChildrenFromBounds: true,
      resize: true,

      // For LayoutBox compatibility
      disabledOpacity: SceneryConstants.DISABLED_OPACITY
    }, providedOptions );

    super();

    this._constraint = new FlowConstraint( this, {
      preferredWidthProperty: this.localPreferredWidthProperty,
      preferredHeightProperty: this.localPreferredHeightProperty,
      minimumWidthProperty: this.localMinimumWidthProperty,
      minimumHeightProperty: this.localMinimumHeightProperty,
      layoutOriginProperty: this.layoutOriginProperty,

      orientation: DEFAULT_OPTIONS.orientation,
      spacing: DEFAULT_OPTIONS.spacing,
      align: DEFAULT_OPTIONS.align,
      stretch: DEFAULT_OPTIONS.stretch,
      excludeInvisible: false // Should be handled by the options mutate below
    } );

    this.onChildInserted = this.onFlowBoxChildInserted.bind( this );
    this.onChildRemoved = this.onFlowBoxChildRemoved.bind( this );
    this.onChildrenReordered = this.onFlowBoxChildrenReordered.bind( this );
    this.onChildrenChanged = this.onFlowBoxChildrenChanged.bind( this );

    this.childInsertedEmitter.addListener( this.onChildInserted );
    this.childRemovedEmitter.addListener( this.onChildRemoved );
    this.childrenReorderedEmitter.addListener( this.onChildrenReordered );
    this.childrenChangedEmitter.addListener( this.onChildrenChanged );

    const nonBoundsOptions = _.omit( options, REQUIRES_BOUNDS_OPTION_KEYS ) as LayoutNodeOptions;
    const boundsOptions = _.pick( options, REQUIRES_BOUNDS_OPTION_KEYS ) as LayoutNodeOptions;

    // Before we do layout, do non-bounds-related changes (in case we have resize:false), and prevent layout for
    // performance gains.
    this._constraint.lock();
    this.mutate( nonBoundsOptions );
    this._constraint.unlock();

    // Update the layout (so that it is done once if we have resize:false)
    this._constraint.updateLayout();

    // After we have our localBounds complete, now we can mutate things that rely on it.
    this.mutate( boundsOptions );

    this.linkLayoutBounds();
  }

  /**
   * Called when a child is inserted.
   */
  private onFlowBoxChildInserted( node: Node, index: number ): void {
    const cell = new FlowCell( this._constraint, node, this._constraint.createLayoutProxy( node ) );
    this._cellMap.set( node, cell );

    this._constraint.insertCell( index, cell );
  }

  /**
   * Called when a child is removed.
   */
  private onFlowBoxChildRemoved( node: Node ): void {

    const cell = this._cellMap.get( node )!;
    assert && assert( cell );

    this._cellMap.delete( node );

    this._constraint.removeCell( cell );

    cell.dispose();
  }

  /**
   * Called when children are rearranged
   */
  private onFlowBoxChildrenReordered( minChangeIndex: number, maxChangeIndex: number ): void {
    this._constraint.reorderCells(
      this._children.slice( minChangeIndex, maxChangeIndex + 1 ).map( node => this._cellMap.get( node )! ),
      minChangeIndex,
      maxChangeIndex
    );
  }

  /**
   * Called on change of children (child added, removed, order changed, etc.)
   */
  private onFlowBoxChildrenChanged(): void {
    this._constraint.updateLayoutAutomatically();
  }

  getCell( node: Node ): FlowCell {
    const result = this._cellMap.get( node )!;
    assert && assert( result );

    return result;
  }

  get orientation(): LayoutOrientation {
    return this._constraint.orientation;
  }

  set orientation( value: LayoutOrientation ) {
    this._constraint.orientation = value;
  }

  get spacing(): number {
    return this._constraint.spacing;
  }

  set spacing( value: number ) {
    this._constraint.spacing = value;
  }

  get lineSpacing(): number {
    return this._constraint.lineSpacing;
  }

  set lineSpacing( value: number ) {
    this._constraint.lineSpacing = value;
  }

  get justify(): HorizontalLayoutJustification | VerticalLayoutJustification {
    return this._constraint.justify;
  }

  set justify( value: HorizontalLayoutJustification | VerticalLayoutJustification ) {
    this._constraint.justify = value;
  }

  get justifyLines(): HorizontalLayoutJustification | VerticalLayoutJustification | null {
    return this._constraint.justifyLines;
  }

  set justifyLines( value: HorizontalLayoutJustification | VerticalLayoutJustification | null ) {
    this._constraint.justifyLines = value;
  }

  get wrap(): boolean {
    return this._constraint.wrap;
  }

  set wrap( value: boolean ) {
    this._constraint.wrap = value;
  }

  get align(): HorizontalLayoutAlign | VerticalLayoutAlign {
    assert && assert( typeof this._constraint.align === 'string' );

    return this._constraint.align!;
  }

  set align( value: HorizontalLayoutAlign | VerticalLayoutAlign ) {
    assert && assert( typeof value === 'string', 'FlowBox align should be a string' );

    this._constraint.align = value;
  }

  get stretch(): boolean {
    assert && assert( typeof this._constraint.stretch === 'boolean' );

    return this._constraint.stretch!;
  }

  set stretch( value: boolean ) {
    this._constraint.stretch = value;
  }

  get grow(): number {
    return this._constraint.grow!;
  }

  set grow( value: number ) {
    this._constraint.grow = value;
  }

  get margin(): number {
    return this._constraint.margin!;
  }

  set margin( value: number ) {
    this._constraint.margin = value;
  }

  get xMargin(): number {
    return this._constraint.xMargin!;
  }

  set xMargin( value: number ) {
    this._constraint.xMargin = value;
  }

  get yMargin(): number {
    return this._constraint.yMargin!;
  }

  set yMargin( value: number ) {
    this._constraint.yMargin = value;
  }

  get leftMargin(): number {
    return this._constraint.leftMargin!;
  }

  set leftMargin( value: number ) {
    this._constraint.leftMargin = value;
  }

  get rightMargin(): number {
    return this._constraint.rightMargin!;
  }

  set rightMargin( value: number ) {
    this._constraint.rightMargin = value;
  }

  get topMargin(): number {
    return this._constraint.topMargin!;
  }

  set topMargin( value: number ) {
    this._constraint.topMargin = value;
  }

  get bottomMargin(): number {
    return this._constraint.bottomMargin!;
  }

  set bottomMargin( value: number ) {
    this._constraint.bottomMargin = value;
  }

  get minContentWidth(): number | null {
    return this._constraint.minContentWidth;
  }

  set minContentWidth( value: number | null ) {
    this._constraint.minContentWidth = value;
  }

  get minContentHeight(): number | null {
    return this._constraint.minContentHeight;
  }

  set minContentHeight( value: number | null ) {
    this._constraint.minContentHeight = value;
  }

  get maxContentWidth(): number | null {
    return this._constraint.maxContentWidth;
  }

  set maxContentWidth( value: number | null ) {
    this._constraint.maxContentWidth = value;
  }

  get maxContentHeight(): number | null {
    return this._constraint.maxContentHeight;
  }

  set maxContentHeight( value: number | null ) {
    this._constraint.maxContentHeight = value;
  }

  /**
   * Releases references
   */
  override dispose(): void {
    this.childInsertedEmitter.removeListener( this.onChildInserted );
    this.childRemovedEmitter.removeListener( this.onChildRemoved );
    this.childrenReorderedEmitter.removeListener( this.onChildrenReordered );
    this.childrenChangedEmitter.removeListener( this.onChildrenChanged );

    // Dispose our cells here. We won't be getting the children-removed listeners fired (we removed them above)
    for ( const cell of this._cellMap.values() ) {
      cell.dispose();
    }

    super.dispose();
  }

  // LayoutBox Compatibility (see the ES5 setters/getters, or the options doc)
  setOrientation( orientation: LayoutOrientation ): this {
    this.orientation = orientation;
    return this;
  }

  getOrientation(): LayoutOrientation { return this.orientation; }

  setSpacing( spacing: number ): this {
    this.spacing = spacing;
    return this;
  }

  getSpacing(): number { return this.spacing; }

  setAlign( align: HorizontalLayoutAlign | VerticalLayoutAlign ): this {
    this.align = align;
    return this;
  }

  getAlign(): HorizontalLayoutAlign | VerticalLayoutAlign { return this.align; }

  setResize( resize: boolean ): this {
    this.resize = resize;
    return this;
  }

  isResize(): boolean { return this.resize; }

  public getHelperNode(): Node {
    const marginsNode = MarginLayoutCell.createHelperNode( this.constraint.displayedCells, this.constraint.layoutBoundsProperty.value, cell => {
      let str = '';

      const internalOrientation = cell.orientation === 'horizontal' ? Orientation.HORIZONTAL : Orientation.VERTICAL;

      str += `align: ${LayoutAlign.internalToAlign( internalOrientation, cell.effectiveAlign )}\n`;
      str += `stretch: ${cell.effectiveStretch}\n`;
      str += `grow: ${cell.effectiveGrow}\n`;

      return str;
    } );

    return marginsNode;
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
FlowBox.prototype._mutatorKeys = [ ...SIZABLE_OPTION_KEYS, ...FLOWBOX_OPTION_KEYS, ...Node.prototype._mutatorKeys ];

// {Object}
FlowBox.DEFAULT_OPTIONS = DEFAULT_OPTIONS;

scenery.register( 'FlowBox', FlowBox );
