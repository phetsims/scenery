// Copyright 2022, University of Colorado Boulder

/**
 * A grid-based layout container. See scenery/docs/layout.html for details (preferably after building scenery).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import assertMutuallyExclusiveOptions from '../../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import optionize from '../../../../phet-core/js/optionize.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import { GRID_CONSTRAINT_OPTION_KEYS, GridCell, GridConstraint, GridConstraintOptions, HEIGHT_SIZABLE_OPTION_KEYS, HorizontalLayoutAlign, LAYOUT_NODE_OPTION_KEYS, LayoutNode, LayoutNodeOptions, Node, NodeOptions, scenery, VerticalLayoutAlign, WIDTH_SIZABLE_OPTION_KEYS } from '../../imports.js';

// GridBox-specific options that can be passed in the constructor or mutate() call.
const GRIDBOX_OPTION_KEYS = [
  ...LAYOUT_NODE_OPTION_KEYS,
  ...GRID_CONSTRAINT_OPTION_KEYS.filter( key => key !== 'excludeInvisible' ),
  'rows',
  'columns',
  'autoRows',
  'autoColumns'
];

// Used for setting/getting rows/columns
type LineArrays = ( Node | null )[][];

type SelfOptions = {
  // Controls whether the GridBox will re-trigger layout automatically after the "first" layout during construction.
  // The GridBox will layout once after processing the options object, but if resize:false, then after that manual
  // layout calls will need to be done (with updateLayout())
  resize?: boolean;

  // Sets the children of the GridBox and positions them using a 2-dimensional array of Node|null (null is a placeholder
  // and does nothing). The first index is treated as a row, and the second is treated as a column, so that:
  //
  //   rows[ row ][ column ] = Node
  //   rows[ y ][ x ] = Node
  //
  // Thus the following will have 2 rows that have 3 columns each:
  //   rows: [ [ a, b, c ], [ d, e, f ] ]
  //
  // NOTE: This will mutate the layoutOptions of the Nodes themselves, and will also wipe out any existing children.
  // NOTE: Don't use this option with either `children` or `columns` also being set
  rows?: LineArrays;

  // Sets the children of the GridBox and positions them using a 2-dimensional array of Node|null (null is a placeholder
  // and does nothing). The first index is treated as a column, and the second is treated as a row, so that:
  //
  //   columns[ column ][ row ] = Node
  //   columns[ x ][ y ] = Node
  //
  // Thus the following will have 2 columns that have 3 rows each:
  //   columns: [ [ a, b, c ], [ d, e, f ] ]
  //
  // NOTE: This will mutate the layoutOptions of the Nodes themselves, and will also wipe out any existing children.
  // NOTE: Don't use this option with either `children` or `rows` also being set
  columns?: LineArrays;

  // When non-null, the cells of this grid will be positioned/sized to be 1x1 cells, filling rows until a column has
  // `autoRows` number of rows, then it will go to the next column. This should generally be used with `children` or
  // adding/removing children in normal ways.
  autoRows?: number | null;

  // When non-null, the cells of this grid will be positioned/sized to be 1x1 cells, filling columns until a row has
  // `autoColumns` number of columns, then it will go to the next row. This should generally be used with `children` or
  // adding/removing children in normal ways.
  autoColumns?: number | null;
} & Omit<GridConstraintOptions, 'excludeInvisible' | 'preferredWidthProperty' | 'preferredHeightProperty' | 'minimumWidthProperty' | 'minimumHeightProperty' | 'layoutOriginProperty'>;

export type GridBoxOptions = SelfOptions & LayoutNodeOptions;

export default class GridBox extends LayoutNode<GridConstraint> {

  private readonly _cellMap: Map<Node, GridCell> = new Map<Node, GridCell>();

  // For handling auto-wrapping features
  private _autoRows: number | null = null;
  private _autoColumns: number | null = null;

  // So we don't kill performance while setting children with autoRows/autoColumns
  private _autoLockCount = 0;

  // Listeners that we'll need to remove
  private readonly onChildInserted: ( node: Node, index: number ) => void;
  private readonly onChildRemoved: ( node: Node ) => void;

  constructor( providedOptions?: GridBoxOptions ) {
    const options = optionize<GridBoxOptions, Omit<SelfOptions, keyof GridConstraintOptions | 'rows' | 'columns' | 'autoRows' | 'autoColumns'>, LayoutNodeOptions>()( {
      // Allow dynamic layout by default, see https://github.com/phetsims/joist/issues/608
      excludeInvisibleChildrenFromBounds: true,

      resize: true
    }, providedOptions );

    super();

    this._constraint = new GridConstraint( this, {
      preferredWidthProperty: this.localPreferredWidthProperty,
      preferredHeightProperty: this.localPreferredHeightProperty,
      minimumWidthProperty: this.localMinimumWidthProperty,
      minimumHeightProperty: this.localMinimumHeightProperty,
      layoutOriginProperty: this.layoutOriginProperty,

      excludeInvisible: false // Should be handled by the options mutate below
    } );

    this.onChildInserted = this.onGridBoxChildInserted.bind( this );
    this.onChildRemoved = this.onGridBoxChildRemoved.bind( this );

    this.childInsertedEmitter.addListener( this.onChildInserted );
    this.childRemovedEmitter.addListener( this.onChildRemoved );

    this.mutate( options );
    this._constraint.updateLayout();

    this.linkLayoutBounds();
  }

  /**
   * Sets the children of the GridBox and adjusts them to be positioned in certain cells. It takes a 2-dimensional array
   * of Node|null (where null is a placeholder that does nothing).
   *
   * For each cell, the first index into the array will be taken as the cell position in the provided orientation. The
   * second index into the array will be taken as the cell position in the OPPOSITE orientation.
   *
   * See GridBox.rows or GridBox.columns for usages and more documentation.
   */
  setLines( orientation: Orientation, lineArrays: LineArrays ): void {
    const children: Node[] = [];

    for ( let i = 0; i < lineArrays.length; i++ ) {
      const lineArray = lineArrays[ i ];
      for ( let j = 0; j < lineArray.length; j++ ) {
        const item = lineArray[ j ];
        if ( item !== null ) {
          children.push( item );
          item.mutateLayoutOptions( {
            [ orientation.coordinate ]: i,
            [ orientation.opposite.coordinate ]: j
          } );
        }
      }
    }

    this.children = children;
  }

  /**
   * Returns the children of the GridBox in a 2-dimensional array of Node|null (where null is a placeholder that does
   * nothing).
   *
   * For each cell, the first index into the array will be taken as the cell position in the provided orientation. The
   * second index into the array will be taken as the cell position in the OPPOSITE orientation.
   *
   * See GridBox.rows or GridBox.columns for usages and more documentation.
   */
  getLines( orientation: Orientation ): LineArrays {
    const lineArrays: LineArrays = [];

    for ( const cell of this._cellMap.values() ) {
      const i = cell.position.get( orientation );
      const j = cell.position.get( orientation.opposite );

      // Ensure we have enough lines
      while ( lineArrays.length < i + 1 ) {
        lineArrays.push( [] );
      }

      // null-pad lines
      while ( lineArrays[ i ].length < j + 1 ) {
        lineArrays[ i ].push( null );
      }

      // Finally the actual node!
      lineArrays[ i ][ j ] = cell.node;
    }

    return lineArrays;
  }

  set rows( lineArrays: LineArrays ) {
    this.setLines( Orientation.VERTICAL, lineArrays );
  }

  get rows(): LineArrays {
    return this.getLines( Orientation.VERTICAL );
  }

  set columns( lineArrays: LineArrays ) {
    this.setLines( Orientation.HORIZONTAL, lineArrays );
  }

  get columns(): LineArrays {
    return this.getLines( Orientation.HORIZONTAL );
  }

  set autoRows( value: number | null ) {
    assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 1 ) );

    if ( this._autoRows !== value ) {
      this._autoRows = value;

      this.updateAutoRows();
    }
  }

  get autoRows(): number | null {
    return this._autoRows;
  }

  set autoColumns( value: number | null ) {
    assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 1 ) );

    if ( this._autoColumns !== value ) {
      this._autoColumns = value;

      this.updateAutoColumns();
    }
  }

  get autoColumns(): number | null {
    return this._autoColumns;
  }

  private updateAutoLines( orientation: Orientation, value: number | null ): void {
    if ( value !== null && this._autoLockCount === 0 ) {
      this.constraint.lock();

      this.children.forEach( ( child, index ) => {
        child.mutateLayoutOptions( {
          [ orientation.coordinate ]: index % value,
          [ orientation.opposite.coordinate ]: Math.floor( index / value ),
          width: 1,
          height: 1
        } );
      } );

      this.constraint.unlock();
      this.constraint.updateLayoutAutomatically();
    }
  }

  private updateAutoRows(): void {
    this.updateAutoLines( Orientation.VERTICAL, this.autoRows );
  }

  private updateAutoColumns(): void {
    this.updateAutoLines( Orientation.HORIZONTAL, this.autoColumns );
  }

  override setChildren( children: Node[] ): this {

    const oldChildren = this.getChildren(); // defensive copy

    // Don't update autoRows/autoColumns settings while setting children, wait until after for performance
    this._autoLockCount++;
    super.setChildren( children );
    this._autoLockCount--;

    if ( !_.isEqual( oldChildren, children ) ) {
      this.updateAutoRows();
      this.updateAutoColumns();
    }

    return this;
  }

  /**
   * Called when a child is inserted.
   */
  private onGridBoxChildInserted( node: Node, index: number ): void {
    const cell = new GridCell( this._constraint, node, this._constraint.createLayoutProxy( node ) );
    this._cellMap.set( node, cell );

    this._constraint.addCell( cell );

    this.updateAutoRows();
    this.updateAutoColumns();
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

    this.updateAutoRows();
    this.updateAutoColumns();
  }

  override mutate( options?: NodeOptions ): this {
    // children can be used with one of autoRows/autoColumns, but otherwise these options are exclusive
    assertMutuallyExclusiveOptions( options, [ 'rows' ], [ 'columns' ], [ 'children', 'autoRows', 'autoColumns' ] );
    assertMutuallyExclusiveOptions( options, [ 'autoRows' ], [ 'autoColumns' ] );

    return super.mutate( options );
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

  get xAlign(): HorizontalLayoutAlign {
    return this._constraint.xAlign!;
  }

  set xAlign( value: HorizontalLayoutAlign ) {
    this._constraint.xAlign = value;
  }

  get yAlign(): VerticalLayoutAlign {
    return this._constraint.yAlign!;
  }

  set yAlign( value: VerticalLayoutAlign ) {
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

  override dispose(): void {

    this.childInsertedEmitter.removeListener( this.onChildInserted );
    this.childRemovedEmitter.removeListener( this.onChildRemoved );

    // Dispose our cells here. We won't be getting the children-removed listeners fired (we removed them above)
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
GridBox.prototype._mutatorKeys = [ ...WIDTH_SIZABLE_OPTION_KEYS, ...HEIGHT_SIZABLE_OPTION_KEYS, ...GRIDBOX_OPTION_KEYS, ...Node.prototype._mutatorKeys ];

scenery.register( 'GridBox', GridBox );
