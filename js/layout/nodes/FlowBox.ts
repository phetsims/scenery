// Copyright 2021-2025, University of Colorado Boulder

/**
 * A vertical/horizontal flow-based layout container.
 *
 * See https://scenerystack.org/learn/scenery-layout/#FlowBox for details
 *
 * FlowBox-only options:
 *   - resize (see https://scenerystack.org/learn/scenery-layout/#FlowBox-resize)
 *   - orientation (see https://scenerystack.org/learn/scenery-layout/#FlowBox-orientation)
 *   - spacing (see https://scenerystack.org/learn/scenery-layout/#FlowBox-spacing)
 *   - lineSpacing (see https://scenerystack.org/learn/scenery-layout/#FlowBox-lineSpacing)
 *   - justify (see https://scenerystack.org/learn/scenery-layout/#FlowBox-justify)
 *   - justifyLines (see https://scenerystack.org/learn/scenery-layout/#FlowBox-justifyLines)
 *   - wrap (see https://scenerystack.org/learn/scenery-layout/#FlowBox-wrap)
 *   - layoutOrigin (see https://scenerystack.org/learn/scenery-layout/#layoutOrigin)
 *   - forward (see https://scenerystack.org/learn/scenery-layout/#FlowBox-forward)
 *
 * FlowBox and layoutOptions options (can be set either in the FlowBox itself, or within its child nodes' layoutOptions):
 *   - align (see https://scenerystack.org/learn/scenery-layout/#FlowBox-align)
 *   - stretch (see https://scenerystack.org/learn/scenery-layout/#FlowBox-stretch)
 *   - grow (see https://scenerystack.org/learn/scenery-layout/#FlowBox-grow)
 *   - cellAlign (see https://scenerystack.org/learn/scenery-layout/#FlowBox-cellAlign)
 *   - margin (see https://scenerystack.org/learn/scenery-layout/#FlowBox-margins)
 *   - xMargin (see https://scenerystack.org/learn/scenery-layout/#FlowBox-margins)
 *   - yMargin (see https://scenerystack.org/learn/scenery-layout/#FlowBox-margins)
 *   - leftMargin (see https://scenerystack.org/learn/scenery-layout/#FlowBox-margins)
 *   - rightMargin (see https://scenerystack.org/learn/scenery-layout/#FlowBox-margins)
 *   - topMargin (see https://scenerystack.org/learn/scenery-layout/#FlowBox-margins)
 *   - bottomMargin (see https://scenerystack.org/learn/scenery-layout/#FlowBox-margins)
 *   - minContentWidth (see https://scenerystack.org/learn/scenery-layout/#FlowBox-minContent)
 *   - minContentHeight (see https://scenerystack.org/learn/scenery-layout/#FlowBox-minContent)
 *   - maxContentWidth (see https://scenerystack.org/learn/scenery-layout/#FlowBox-maxContent)
 *   - maxContentHeight (see https://scenerystack.org/learn/scenery-layout/#FlowBox-maxContent)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import optionize from '../../../../phet-core/js/optionize.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import { FLOW_CONSTRAINT_OPTION_KEYS } from '../../layout/constraints/FlowConstraint.js';
import FlowCell from '../../layout/constraints/FlowCell.js';
import FlowConstraint from '../../layout/constraints/FlowConstraint.js';
import type { FlowConstraintOptions } from '../../layout/constraints/FlowConstraint.js';
import { HorizontalLayoutAlign, VerticalLayoutAlign } from '../../layout/LayoutAlign.js';
import { HorizontalLayoutJustification, VerticalLayoutJustification } from '../../layout/LayoutJustification.js';
import { LAYOUT_NODE_OPTION_KEYS } from '../../layout/nodes/LayoutNode.js';
import LayoutAlign from '../../layout/LayoutAlign.js';
import LayoutNode from '../../layout/nodes/LayoutNode.js';
import type { LayoutNodeOptions } from '../../layout/nodes/LayoutNode.js';
import { LayoutOrientation } from '../../layout/LayoutOrientation.js';
import MarginLayoutCell from '../../layout/constraints/MarginLayoutCell.js';
import Node, { REQUIRES_BOUNDS_OPTION_KEYS } from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import SceneryConstants from '../../SceneryConstants.js';
import { SIZABLE_OPTION_KEYS } from '../../layout/Sizable.js';
import TReadOnlyProperty from '../../../../axon/js/TReadOnlyProperty.js';
import TinyForwardingProperty from '../../../../axon/js/TinyForwardingProperty.js';
import TProperty from '../../../../axon/js/TProperty.js';

// FlowBox-specific options that can be passed in the constructor or mutate() call.
const FLOWBOX_OPTION_KEYS = [
  ...LAYOUT_NODE_OPTION_KEYS,
  ...FLOW_CONSTRAINT_OPTION_KEYS.filter( key => key !== 'excludeInvisible' ),
  'forward',
  'forwardProperty'
];

const DEFAULT_OPTIONS = {
  orientation: 'horizontal',
  spacing: 0,
  align: 'center',
  stretch: false,
  forward: true
} as const;

type ExcludeFlowConstraintOptions = 'excludeInvisible' | 'preferredWidthProperty' | 'preferredHeightProperty' | 'minimumWidthProperty' | 'minimumHeightProperty' | 'layoutOriginProperty';
type SelfOptions = {
  // Controls whether the FlowBox will re-trigger layout automatically after the "first" layout during construction.
  // The FlowBox will layout once after processing the options object, but if resize:false, then after that manual
  // layout calls will need to be done (with updateLayout())
  resize?: boolean;

  // If true, the nodes in this FlowBox will be ordered in a forward order.
  // If false, the nodes will appear reversed in order.
  forward?: boolean;

  // Like forward, but allows specifying a boolean Property to forward the order
  // Notably, for HBoxes this is useful to pass in localeProperty.
  forwardProperty?: TReadOnlyProperty<boolean> | null;
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

  // Whether this FlowBox will have its elements forward or reversed (stored as a
  // forwardable Property so localeProperty could be used).
  private readonly _forwardProperty: TinyForwardingProperty<boolean>;

  public constructor( providedOptions?: FlowBoxOptions ) {
    const options = optionize<FlowBoxOptions, StrictOmit<SelfOptions, Exclude<keyof FlowConstraintOptions, ExcludeFlowConstraintOptions>>, LayoutNodeOptions>()( {
      // Allow dynamic layout by default, see https://github.com/phetsims/joist/issues/608
      excludeInvisibleChildrenFromBounds: true,
      resize: true,

      // For LayoutBox compatibility
      disabledOpacity: SceneryConstants.DISABLED_OPACITY,

      forward: DEFAULT_OPTIONS.forward,
      forwardProperty: null
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

    this._forwardProperty = new TinyForwardingProperty<boolean>( DEFAULT_OPTIONS.forward, false, this.onFlowBoxForwardReverseChanged.bind( this ) );

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
  protected onFlowBoxChildInserted( node: Node, index: number ): void {
    const cell = new FlowCell( this._constraint, node, this._constraint.createLayoutProxy( node ) );
    this._cellMap.set( node, cell );

    if ( this.forward ) {
      this._constraint.insertCell( index, cell );
    }
    else {
      const reverseIndex = this._constraint.getCellQuantity() - index;

      this._constraint.insertCell( reverseIndex, cell );
    }
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
    const nodes = this._children.slice( minChangeIndex, maxChangeIndex + 1 );

    if ( !this.forward ) {
      nodes.reverse();
    }

    this._constraint.reorderCells(
      nodes.map( node => this._cellMap.get( node )! ),
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

  /**
   * Called when our `forward` value changes.
   */
  private onFlowBoxForwardReverseChanged(): void {
    this.onFlowBoxChildrenReordered( 0, this._children.length - 1 );
    this.onFlowBoxChildrenChanged();
  }

  public getCell( node: Node ): FlowCell {
    const result = this._cellMap.get( node )!;
    assert && assert( result );

    return result;
  }

  public get orientation(): LayoutOrientation {
    return this._constraint.orientation;
  }

  public set orientation( value: LayoutOrientation ) {
    this._constraint.orientation = value;
  }

  public get spacing(): number {
    return this._constraint.spacing;
  }

  public set spacing( value: number ) {
    this._constraint.spacing = value;
  }

  public get lineSpacing(): number {
    return this._constraint.lineSpacing;
  }

  public set lineSpacing( value: number ) {
    this._constraint.lineSpacing = value;
  }

  public get justify(): HorizontalLayoutJustification | VerticalLayoutJustification {
    return this._constraint.justify;
  }

  public set justify( value: HorizontalLayoutJustification | VerticalLayoutJustification ) {
    this._constraint.justify = value;
  }

  public get justifyLines(): HorizontalLayoutJustification | VerticalLayoutJustification | null {
    return this._constraint.justifyLines;
  }

  public set justifyLines( value: HorizontalLayoutJustification | VerticalLayoutJustification | null ) {
    this._constraint.justifyLines = value;
  }

  public get wrap(): boolean {
    return this._constraint.wrap;
  }

  public set wrap( value: boolean ) {
    this._constraint.wrap = value;
  }

  public get align(): HorizontalLayoutAlign | VerticalLayoutAlign {
    assert && assert( typeof this._constraint.align === 'string' );

    return this._constraint.align!;
  }

  public set align( value: HorizontalLayoutAlign | VerticalLayoutAlign ) {
    this._constraint.align = value;
  }

  public get stretch(): boolean {
    assert && assert( typeof this._constraint.stretch === 'boolean' );

    return this._constraint.stretch!;
  }

  public set stretch( value: boolean ) {
    this._constraint.stretch = value;
  }

  public get grow(): number {
    return this._constraint.grow!;
  }

  public set grow( value: number ) {
    this._constraint.grow = value;
  }

  public get margin(): number {
    return this._constraint.margin!;
  }

  public set margin( value: number ) {
    this._constraint.margin = value;
  }

  public get xMargin(): number {
    return this._constraint.xMargin!;
  }

  public set xMargin( value: number ) {
    this._constraint.xMargin = value;
  }

  public get yMargin(): number {
    return this._constraint.yMargin!;
  }

  public set yMargin( value: number ) {
    this._constraint.yMargin = value;
  }

  public get leftMargin(): number {
    return this._constraint.leftMargin!;
  }

  public set leftMargin( value: number ) {
    this._constraint.leftMargin = value;
  }

  public get rightMargin(): number {
    return this._constraint.rightMargin!;
  }

  public set rightMargin( value: number ) {
    this._constraint.rightMargin = value;
  }

  public get topMargin(): number {
    return this._constraint.topMargin!;
  }

  public set topMargin( value: number ) {
    this._constraint.topMargin = value;
  }

  public get bottomMargin(): number {
    return this._constraint.bottomMargin!;
  }

  public set bottomMargin( value: number ) {
    this._constraint.bottomMargin = value;
  }

  public get minContentWidth(): number | null {
    return this._constraint.minContentWidth;
  }

  public set minContentWidth( value: number | null ) {
    this._constraint.minContentWidth = value;
  }

  public get minContentHeight(): number | null {
    return this._constraint.minContentHeight;
  }

  public set minContentHeight( value: number | null ) {
    this._constraint.minContentHeight = value;
  }

  public get maxContentWidth(): number | null {
    return this._constraint.maxContentWidth;
  }

  public set maxContentWidth( value: number | null ) {
    this._constraint.maxContentWidth = value;
  }

  public get maxContentHeight(): number | null {
    return this._constraint.maxContentHeight;
  }

  public set maxContentHeight( value: number | null ) {
    this._constraint.maxContentHeight = value;
  }

  /**
   * Sets what Property our forwardProperty is backed by
   */
  public setForwardProperty( newTarget: TReadOnlyProperty<boolean> | null ): this {
    return this._forwardProperty.setTargetProperty( newTarget, this, null );
  }

  /**
   * See setForwardProperty() for more information
   */
  public set forwardProperty( property: TReadOnlyProperty<boolean> | null ) {
    this.setForwardProperty( property );
  }

  /**
   * See getForwardProperty() for more information
   */
  public get forwardProperty(): TProperty<boolean> {
    return this.getForwardProperty();
  }

  /**
   * Get this Node's forwardProperty. Note! This is not the reciprocal of setForwardProperty. Node.prototype._forwardProperty
   * is a TinyForwardingProperty, and is set up to listen to changes from the forwardProperty provided by
   * setForwardProperty(), but the underlying reference does not change. This means the following:
   *     * const myNode = new Node();
   * const forwardProperty = new Property( false );
   * myNode.setForwardProperty( forwardProperty )
   * => myNode.getForwardProperty() !== forwardProperty (!!!!!!)
   *
   * Please use this with caution. See setForwardProperty() for more information.
   */
  public getForwardProperty(): TProperty<boolean> {
    return this._forwardProperty;
  }

  /**
   * Sets whether the nodes in this FlowBox are ordered in a forward order.
   * If false, the nodes will appear reversed in the layout.
   */
  public setForward( forward: boolean ): this {
    this.forwardProperty.set( forward );
    return this;
  }

  /**
   * See setForward() for more information
   */
  public set forward( value: boolean ) {
    this.setForward( value );
  }

  /**
   * See isForward() for more information
   */
  public get forward(): boolean {
    return this.isForward();
  }

  /**
   * Returns whether this Node is forward.
   */
  public isForward(): boolean {
    return this.forwardProperty.value;
  }

  /**
   * Releases references
   */
  public override dispose(): void {

    // Lock our layout forever
    this._constraint.lock();

    this.childInsertedEmitter.removeListener( this.onChildInserted );
    this.childRemovedEmitter.removeListener( this.onChildRemoved );
    this.childrenReorderedEmitter.removeListener( this.onChildrenReordered );
    this.childrenChangedEmitter.removeListener( this.onChildrenChanged );

    this._forwardProperty.dispose();

    // Dispose our cells here. We won't be getting the children-removed listeners fired (we removed them above)
    for ( const cell of this._cellMap.values() ) {
      cell.dispose();
    }

    super.dispose();
  }

  // LayoutBox Compatibility (see the ES5 setters/getters, or the options doc)
  public setOrientation( orientation: LayoutOrientation ): this {
    this.orientation = orientation;
    return this;
  }

  public getOrientation(): LayoutOrientation { return this.orientation; }

  public setSpacing( spacing: number ): this {
    this.spacing = spacing;
    return this;
  }

  public getSpacing(): number { return this.spacing; }

  public setAlign( align: HorizontalLayoutAlign | VerticalLayoutAlign ): this {
    this.align = align;
    return this;
  }

  public getAlign(): HorizontalLayoutAlign | VerticalLayoutAlign { return this.align; }

  public setResize( resize: boolean ): this {
    this.resize = resize;
    return this;
  }

  public isResize(): boolean { return this.resize; }

  public getHelperNode(): Node {
    const marginsNode = MarginLayoutCell.createHelperNode( this.constraint.displayedCells, this.constraint.layoutBoundsProperty.value, cell => {
      let str = '';

      const internalOrientation = Orientation.fromLayoutOrientation( cell.orientation );

      str += `align: ${LayoutAlign.internalToAlign( internalOrientation, cell.effectiveAlign )}\n`;
      str += `stretch: ${cell.effectiveStretch}\n`;
      str += `grow: ${cell.effectiveGrow}\n`;

      return str;
    } );

    return marginsNode;
  }

  public override mutate( options?: FlowBoxOptions ): this {

    if ( !options ) {
      return this;
    }

    if ( assert && options.hasOwnProperty( 'forward' ) && options.hasOwnProperty( 'forwardProperty' ) && options.forwardProperty !== null ) {
      assert && assert( options.forwardProperty!.value === options.forward, 'If both forward and forwardProperty are provided, then values should match' );
    }

    return super.mutate( options );
  }

  public static readonly DEFAULT_FLOW_BOX_OPTIONS = DEFAULT_OPTIONS;
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
FlowBox.prototype._mutatorKeys = [ ...SIZABLE_OPTION_KEYS, ...FLOWBOX_OPTION_KEYS, ...Node.prototype._mutatorKeys ];

scenery.register( 'FlowBox', FlowBox );