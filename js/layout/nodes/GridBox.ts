// Copyright 2022, University of Colorado Boulder

/**
 * A grid-based layout container.
 *
 * See https://phetsims.github.io/scenery/doc/layout#GridBox for details
 *
 * GridBox-only options:
 *   - rows (see https://phetsims.github.io/scenery/doc/layout#GridBox-rows)
 *   - columns (see https://phetsims.github.io/scenery/doc/layout#GridBox-columns)
 *   - autoRows (see https://phetsims.github.io/scenery/doc/layout#GridBox-autoLines)
 *   - autoColumns (see https://phetsims.github.io/scenery/doc/layout#GridBox-autoLines)
 *   - resize (see https://phetsims.github.io/scenery/doc/layout#GridBox-resize)
 *   - spacing (see https://phetsims.github.io/scenery/doc/layout#GridBox-spacing)
 *   - xSpacing (see https://phetsims.github.io/scenery/doc/layout#GridBox-spacing)
 *   - ySpacing (see https://phetsims.github.io/scenery/doc/layout#GridBox-spacing)
 *   - layoutOrigin (see https://phetsims.github.io/scenery/doc/layout#layoutOrigin)
 *
 * GridBox and layoutOptions options:
 *   - xAlign (see https://phetsims.github.io/scenery/doc/layout#GridBox-align)
 *   - yAlign (see https://phetsims.github.io/scenery/doc/layout#GridBox-align)
 *   - stretch (see https://phetsims.github.io/scenery/doc/layout#GridBox-stretch)
 *   - xStretch (see https://phetsims.github.io/scenery/doc/layout#GridBox-stretch)
 *   - yStretch (see https://phetsims.github.io/scenery/doc/layout#GridBox-stretch)
 *   - grow (see https://phetsims.github.io/scenery/doc/layout#GridBox-grow)
 *   - xGrow (see https://phetsims.github.io/scenery/doc/layout#GridBox-grow)
 *   - yGrow (see https://phetsims.github.io/scenery/doc/layout#GridBox-grow)
 *   - margin (see https://phetsims.github.io/scenery/doc/layout#GridBox-margins)
 *   - xMargin (see https://phetsims.github.io/scenery/doc/layout#GridBox-margins)
 *   - yMargin (see https://phetsims.github.io/scenery/doc/layout#GridBox-margins)
 *   - leftMargin (see https://phetsims.github.io/scenery/doc/layout#GridBox-margins)
 *   - rightMargin (see https://phetsims.github.io/scenery/doc/layout#GridBox-margins)
 *   - topMargin (see https://phetsims.github.io/scenery/doc/layout#GridBox-margins)
 *   - bottomMargin (see https://phetsims.github.io/scenery/doc/layout#GridBox-margins)
 *   - minContentWidth (see https://phetsims.github.io/scenery/doc/layout#GridBox-minContent)
 *   - minContentHeight (see https://phetsims.github.io/scenery/doc/layout#GridBox-minContent)
 *   - maxContentWidth (see https://phetsims.github.io/scenery/doc/layout#GridBox-maxContent)
 *   - maxContentHeight (see https://phetsims.github.io/scenery/doc/layout#GridBox-maxContent)
 *
 * layoutOptions-only options:
 *   - x (see https://phetsims.github.io/scenery/doc/layout#GridBox-layoutOptions-location)
 *   - y (see https://phetsims.github.io/scenery/doc/layout#GridBox-layoutOptions-location)
 *   - width (see https://phetsims.github.io/scenery/doc/layout#GridBox-layoutOptions-size)
 *   - height (see https://phetsims.github.io/scenery/doc/layout#GridBox-layoutOptions-size)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import assertMutuallyExclusiveOptions from '../../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import optionize from '../../../../phet-core/js/optionize.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import { GRID_CONSTRAINT_OPTION_KEYS, GridCell, GridConstraint, GridConstraintOptions, HorizontalLayoutAlign, LAYOUT_NODE_OPTION_KEYS, LayoutNode, LayoutNodeOptions, Node, NodeOptions, REQUIRES_BOUNDS_OPTION_KEYS, scenery, SIZABLE_OPTION_KEYS, VerticalLayoutAlign } from '../../imports.js';

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
type LineArray = ( Node | null )[];
type LineArrays = LineArray[];

type GridConstraintExcludedOptions = 'excludeInvisible' | 'preferredWidthProperty' | 'preferredHeightProperty' | 'minimumWidthProperty' | 'minimumHeightProperty' | 'layoutOriginProperty';
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
} & StrictOmit<GridConstraintOptions, GridConstraintExcludedOptions>;

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
    const options = optionize<GridBoxOptions, StrictOmit<SelfOptions, Exclude<keyof GridConstraintOptions, GridConstraintExcludedOptions> | 'rows' | 'columns' | 'autoRows' | 'autoColumns'>,
      LayoutNodeOptions>()( {
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

    const nonBoundsOptions = _.omit( options, REQUIRES_BOUNDS_OPTION_KEYS ) as LayoutNodeOptions;
    const boundsOptions = _.pick( options, REQUIRES_BOUNDS_OPTION_KEYS ) as LayoutNodeOptions;

    // Before we layout, do non-bounds-related changes (in case we have resize:false), and prevent layout for
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
   * See GridBox.rows or GridBox.columns for usages
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

  /**
   * Sets the children of the GridBox by specifying a two-dimensional array of Nodes (or null values as spacers).
   * The inner arrays will be the rows of the grid.
   * Mutates layoutOptions of the provided Nodes. See setLines() for more documentation.
   */
  set rows( lineArrays: LineArrays ) {
    this.setLines( Orientation.VERTICAL, lineArrays );
  }

  /**
   * Returns a two-dimensional array of the child Nodes (with null as a spacer) where the inner arrays are the rows.
   */
  get rows(): LineArrays {
    return this.getLines( Orientation.VERTICAL );
  }

  /**
   * Sets the children of the GridBox by specifying a two-dimensional array of Nodes (or null values as spacers).
   * The inner arrays will be the columns of the grid.
   * * Mutates layoutOptions of the provided Nodes. See setLines() for more documentation.
   */
  set columns( lineArrays: LineArrays ) {
    this.setLines( Orientation.HORIZONTAL, lineArrays );
  }

  /**
   * Returns a two-dimensional array of the child Nodes (with null as a spacer) where the inner arrays are the columns.
   */
  get columns(): LineArrays {
    return this.getLines( Orientation.HORIZONTAL );
  }

  /**
   * Returns the Node at a specific row/column intersection (or null if there are none)
   */
  getNodeAt( row: number, column: number ): Node | null {
    const cell = this.constraint.getCell( row, column );

    return cell ? cell.node : null;
  }

  /**
   * Returns the row index of a child Node (or if it spans multiple rows, the first row)
   */
  getRowOfNode( node: Node ): number {
    assert && assert( this.children.includes( node ) );

    return this.constraint.getCellFromNode( node )!.position.vertical;
  }

  /**
   * Returns the column index of a child Node (or if it spans multiple columns, the first row)
   */
  getColumnOfNode( node: Node ): number {
    assert && assert( this.children.includes( node ) );

    return this.constraint.getCellFromNode( node )!.position.horizontal;
  }

  /**
   * Returns all the Nodes in a given row (by index)
   */
  getNodesInRow( index: number ): Node[] {
    return this.constraint.getCells( Orientation.VERTICAL, index ).map( cell => cell.node );
  }

  /**
   * Returns all the Nodes in a given column (by index)
   */
  getNodesInColumn( index: number ): Node[] {
    return this.constraint.getCells( Orientation.HORIZONTAL, index ).map( cell => cell.node );
  }

  /**
   * Adds an array of child Nodes (with null allowed as empty spacers) at the bottom of all existing rows.
   */
  addRow( row: LineArray ): this {

    this.rows = [ ...this.rows, row ];

    return this;
  }

  /**
   * Adds an array of child Nodes (with null allowed as empty spacers) at the right of all existing columns.
   */
  addColumn( column: LineArray ): this {

    this.columns = [ ...this.columns, column ];

    return this;
  }

  /**
   * Inserts a row of child Nodes at a given row index (see addRow for more information)
   */
  insertRow( index: number, row: LineArray ): this {

    this.rows = [ ...this.rows.slice( 0, index ), row, ...this.rows.slice( index ) ];

    return this;
  }

  /**
   * Inserts a column of child Nodes at a given column index (see addColumn for more information)
   */
  insertColumn( index: number, column: LineArray ): this {

    this.columns = [ ...this.columns.slice( 0, index ), column, ...this.columns.slice( index ) ];

    return this;
  }

  /**
   * Removes all child Nodes in a given row
   */
  removeRow( index: number ): this {

    this.rows = [ ...this.rows.slice( 0, index ), ...this.rows.slice( index + 1 ) ];

    return this;
  }

  /**
   * Removes all child Nodes in a given column
   */
  removeColumn( index: number ): this {

    this.columns = [ ...this.columns.slice( 0, index ), ...this.columns.slice( index + 1 ) ];

    return this;
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

  // Used for autoRows/autoColumns
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

  get stretch(): boolean {
    return this._constraint.stretch!;
  }

  set stretch( value: boolean ) {
    this._constraint.stretch = value;
  }

  get xStretch(): boolean {
    return this._constraint.xStretch!;
  }

  set xStretch( value: boolean ) {
    this._constraint.xStretch = value;
  }

  get yStretch(): boolean {
    return this._constraint.yStretch!;
  }

  set yStretch( value: boolean ) {
    this._constraint.yStretch = value;
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
GridBox.prototype._mutatorKeys = [ ...SIZABLE_OPTION_KEYS, ...GRIDBOX_OPTION_KEYS, ...Node.prototype._mutatorKeys ];

scenery.register( 'GridBox', GridBox );
