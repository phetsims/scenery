// Copyright 2022, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import optionize from '../../../phet-core/js/optionize.js';
import { GRID_CONSTRAINT_OPTION_KEYS, GridCell, GridConstraint, GridConstraintOptions, GridHorizontalAlign, GridVerticalAlign, HeightSizable, HeightSizableSelfOptions, ILayoutOptions, Node, NodeOptions, scenery, WidthSizable, WidthSizableSelfOptions, WIDTH_SIZABLE_OPTION_KEYS, HEIGHT_SIZABLE_OPTION_KEYS } from '../imports.js';

// GridBox-specific options that can be passed in the constructor or mutate() call.
const GRIDBOX_OPTION_KEYS = [
  'resize' // {boolean} - Whether we should update the layout when children change, see setResize for documentation
].concat( GRID_CONSTRAINT_OPTION_KEYS ).filter( key => key !== 'excludeInvisible' );

const DEFAULT_OPTIONS = {
  resize: true // TODO: how is resize working
} as const;

type SelfOptions = {
  excludeInvisibleChildrenFromBounds?: boolean;
  resize?: boolean;
} & Omit<GridConstraintOptions, 'excludeInvisible'>;

type SuperType = NodeOptions & WidthSizableSelfOptions & HeightSizableSelfOptions;

export type GridBoxOptions = SelfOptions & SuperType;

export default class GridBox extends WidthSizable( HeightSizable( Node ) ) {

  private readonly _cellMap: Map<Node, GridCell>;
  private readonly _constraint: GridConstraint;

  // For handling the shortcut-style API
  private _nextX: number;
  private _nextY: number;

  constructor( providedOptions?: GridBoxOptions ) {
    const options = optionize<GridBoxOptions, Omit<SelfOptions, keyof GridConstraintOptions>, SuperType>()( {
      // Allow dynamic layout by default, see https://github.com/phetsims/joist/issues/608
      excludeInvisibleChildrenFromBounds: true,

      resize: true
    }, providedOptions );

    super();

    this._cellMap = new Map();
    this._constraint = new GridConstraint( this, {
      preferredWidthProperty: this.preferredWidthProperty,
      preferredHeightProperty: this.preferredHeightProperty,
      minimumWidthProperty: this.minimumWidthProperty,
      minimumHeightProperty: this.minimumHeightProperty,

      excludeInvisible: false // Should be handled by the options mutate above
    } );

    this._nextX = 0;
    this._nextY = 0;

    this.childInsertedEmitter.addListener( this.onGridBoxChildInserted.bind( this ) );
    this.childRemovedEmitter.addListener( this.onGridBoxChildRemoved.bind( this ) );

    this.mutate( options );
    this._constraint.updateLayout();

    // Adjust the localBounds to be the laid-out area
    this._constraint.layoutBoundsProperty.link( layoutBounds => {
      this.localBounds = layoutBounds;
    } );
  }

  override setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds: boolean ): void {
    super.setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds );

    this._constraint.excludeInvisible = excludeInvisibleChildrenFromBounds;
  }

  /**
   * Called when a child is inserted.
   */
  private onGridBoxChildInserted( node: Node, index: number ): void {
    let layoutOptions = node.layoutOptions;

    if ( !layoutOptions || ( typeof layoutOptions.x !== 'number' && typeof layoutOptions.y !== 'number' ) ) {
      layoutOptions = optionize<ILayoutOptions, {}, ILayoutOptions>()( {
        x: this._nextX,
        y: this._nextY
      }, layoutOptions || undefined );
    }

    if ( layoutOptions!.wrap ) {
      // TODO: how to handle wrapping with larger spans?
      this._nextX = 0;
      this._nextY++;
    }
    else {
      this._nextX = layoutOptions.x! + ( layoutOptions.width || 1 );
      this._nextY = layoutOptions.y!;
    }

    // Go to the next spot
    while ( this._constraint.getCell( this._nextY, this._nextX ) ) {
      this._nextX++;
    }

    const cell = new GridCell( this._constraint, node, this._constraint.createLayoutProxy( node ) );
    this._cellMap.set( node, cell );

    this._constraint.addCell( cell );
  }

  /**
   * Called when a child is removed.
   */
  private onGridBoxChildRemoved( node: Node ): void {

    const cell = this._cellMap.get( node )!;
    assert && assert( cell );

    this._cellMap.delete( node );

    this._constraint.removeCell( cell );

    cell.dispose();
  }

  get resize(): boolean {
    return this._constraint.enabled;
  }

  set resize( value: boolean ) {
    this._constraint.enabled = value;
  }

  get spacing(): number | number[] {
    return this._constraint.spacing;
  }

  set spacing( value: number | number[] ) {
    this._constraint.spacing = value;
  }

  get xSpacing(): number | number[] {
    return this._constraint.xSpacing;
  }

  set xSpacing( value: number | number[] ) {
    this._constraint.xSpacing = value;
  }

  get ySpacing(): number | number[] {
    return this._constraint.ySpacing;
  }

  set ySpacing( value: number | number[] ) {
    this._constraint.ySpacing = value;
  }

  get xAlign(): GridHorizontalAlign {
    return this._constraint.xAlign!;
  }

  set xAlign( value: GridHorizontalAlign ) {
    this._constraint.xAlign = value;
  }

  get yAlign(): GridVerticalAlign {
    return this._constraint.yAlign!;
  }

  set yAlign( value: GridVerticalAlign ) {
    this._constraint.yAlign = value;
  }

  get grow(): number {
    return this._constraint.grow!;
  }

  set grow( value: number ) {
    this._constraint.grow = value;
  }

  get xGrow(): number {
    return this._constraint.xGrow!;
  }

  set xGrow( value: number ) {
    this._constraint.xGrow = value;
  }

  get yGrow(): number {
    return this._constraint.yGrow!;
  }

  set yGrow( value: number ) {
    this._constraint.yGrow = value;
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
   * Manual access to the constraint
   */
  get constraint(): GridConstraint {
    return this._constraint;
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
GridBox.prototype._mutatorKeys = [ ...WIDTH_SIZABLE_OPTION_KEYS, ...HEIGHT_SIZABLE_OPTION_KEYS, ...GRIDBOX_OPTION_KEYS, ...Node.prototype._mutatorKeys ];

// {Object}
GridBox.DEFAULT_OPTIONS = DEFAULT_OPTIONS;

scenery.register( 'GridBox', GridBox );
