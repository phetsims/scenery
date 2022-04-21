// Copyright 2021-2022, University of Colorado Boulder

/**
 * A vertical/horizontal flow-based layout container. TODO: more docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import { scenery, Node, FlowCell, FlowConstraint, WidthSizable, HeightSizable, FLOW_CONSTRAINT_OPTION_KEYS, NodeOptions, FlowConstraintOptions, WidthSizableSelfOptions, HeightSizableSelfOptions, FlowConfigurableOptions, FlowOrientation, FlowHorizontalJustifys, FlowVerticalJustifys, FlowHorizontalAlign, FlowVerticalAlign } from '../imports.js';

// FlowBox-specific options that can be passed in the constructor or mutate() call.
const FLOWBOX_OPTION_KEYS = [
  'resize' // {boolean} - Whether we should update the layout when children change, see setResize for documentation
].concat( FLOW_CONSTRAINT_OPTION_KEYS ).filter( key => key !== 'excludeInvisible' );

const DEFAULT_OPTIONS = {
  orientation: 'horizontal',
  spacing: 0,
  align: 'center'
} as const;

type SelfOptions = {
  excludeInvisibleChildrenFromBounds?: boolean;
  resize?: boolean;
} & Omit<FlowConstraintOptions, 'excludeInvisible'>;

export type FlowBoxOptions = SelfOptions & NodeOptions & WidthSizableSelfOptions & HeightSizableSelfOptions;

export default class FlowBox extends WidthSizable( HeightSizable( Node ) ) {

  private _constraint: FlowConstraint;
  private _cellMap: Map<Node, FlowCell>;

  constructor( providedOptions?: FlowBoxOptions ) {
    // TODO: optionize
    const options = merge( {
      // Allow dynamic layout by default, see https://github.com/phetsims/joist/issues/608
      excludeInvisibleChildrenFromBounds: true,
      resize: true
    }, providedOptions );

    super();

    this._constraint = new FlowConstraint( this, {
      preferredWidthProperty: this.preferredWidthProperty,
      preferredHeightProperty: this.preferredHeightProperty,
      minimumWidthProperty: this.minimumWidthProperty,
      minimumHeightProperty: this.minimumHeightProperty,

      orientation: DEFAULT_OPTIONS.orientation,
      spacing: DEFAULT_OPTIONS.spacing,
      align: DEFAULT_OPTIONS.align,
      excludeInvisible: false // Should be handled by the options mutate above
    } );

    this._cellMap = new Map<Node, FlowCell>();

    this.childInsertedEmitter.addListener( this.onFlowBoxChildInserted.bind( this ) );
    this.childRemovedEmitter.addListener( this.onFlowBoxChildRemoved.bind( this ) );
    this.childrenReorderedEmitter.addListener( this.onFlowBoxChildrenReordered.bind( this ) );
    this.childrenChangedEmitter.addListener( this.onFlowBoxChildrenChanged.bind( this ) );

    this.mutate( options );
    this._constraint.updateLayout();

    // Adjust the localBounds to be the laid-out area
    this._constraint.layoutBoundsProperty.link( layoutBounds => {
      this.localBounds = layoutBounds;
    } );
  }

  override setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds: boolean ) {
    super.setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds );

    this._constraint.excludeInvisible = excludeInvisibleChildrenFromBounds;
  }

  /**
   * Called when a child is inserted.
   */
  private onFlowBoxChildInserted( node: Node, index: number ) {
    const cell = new FlowCell( this._constraint, node, node.layoutOptions as FlowConfigurableOptions );
    this._cellMap.set( node, cell );

    this._constraint.insertCell( index, cell );
  }

  /**
   * Called when a child is removed.
   */
  private onFlowBoxChildRemoved( node: Node ) {

    const cell = this._cellMap.get( node )!;
    assert && assert( cell );

    this._cellMap.delete( node );

    this._constraint.removeCell( cell );

    cell.dispose();
  }

  /**
   * Called when children are rearranged
   */
  private onFlowBoxChildrenReordered( minChangeIndex: number, maxChangeIndex: number ) {
    this._constraint.reorderCells(
      this._children.slice( minChangeIndex, maxChangeIndex + 1 ).map( node => this._cellMap.get( node )! ),
      minChangeIndex,
      maxChangeIndex
    );
  }

  /**
   * Called on change of children (child added, removed, order changed, etc.)
   */
  private onFlowBoxChildrenChanged() {
    this._constraint.updateLayoutAutomatically();
  }

  /**
   * Sets the children of the Node to be equivalent to the passed-in array of Nodes. Does this by removing all current
   * children, and adding in children from the array.
   *
   * Overridden so we can group together setChildren() and only update layout (a) at the end, and (b) if there
   * are changes.
   */
  override setChildren( children: Node[] ): this {
    // If the layout is already locked, we need to bail and only call Node's setChildren.
    if ( this._constraint.isLocked ) {
      return super.setChildren( children );
    }

    const oldChildren = this.getChildren(); // defensive copy

    // Lock layout while the children are removed and added
    this._constraint.lock();
    super.setChildren( children );
    this._constraint.unlock();

    // Determine if the children array has changed. We'll gain a performance benefit by not triggering layout when
    // the children haven't changed.
    if ( !_.isEqual( oldChildren, children ) ) {
      this._constraint.updateLayoutAutomatically();
    }

    return this;
  }

  getCell( node: Node ): FlowCell {
    const result = this._cellMap.get( node )!;
    assert && assert( result );

    return result;
  }

  get resize(): boolean {
    return this._constraint.enabled;
  }

  set resize( value: boolean ) {
    this._constraint.enabled = value;
  }

  get orientation(): FlowOrientation {
    return this._constraint.orientation;
  }

  set orientation( value: FlowOrientation ) {
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

  get justify(): FlowHorizontalJustifys | FlowVerticalJustifys {
    return this._constraint.justify;
  }

  set justify( value: FlowHorizontalJustifys | FlowVerticalJustifys ) {
    this._constraint.justify = value;
  }

  get wrap(): boolean {
    return this._constraint.wrap;
  }

  set wrap( value: boolean ) {
    this._constraint.wrap = value;
  }

  get align(): FlowHorizontalAlign | FlowVerticalAlign {
    assert && assert( typeof this._constraint.align === 'string' );

    return this._constraint.align!;
  }

  set align( value: FlowHorizontalAlign | FlowVerticalAlign ) {
    assert && assert( typeof value === 'string', 'FlowBox align should be a string' );

    this._constraint.align = value;
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
  override dispose() {
    this._constraint.dispose();

    for ( const cell of this._cellMap.values() ) {
      cell.dispose();
    }

    super.dispose();
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
FlowBox.prototype._mutatorKeys = WidthSizable( Node ).prototype._mutatorKeys.concat( HeightSizable( Node ).prototype._mutatorKeys ).concat( FLOWBOX_OPTION_KEYS );

// {Object}
FlowBox.DEFAULT_OPTIONS = DEFAULT_OPTIONS;

scenery.register( 'FlowBox', FlowBox );
