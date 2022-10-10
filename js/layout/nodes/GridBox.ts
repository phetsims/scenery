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
 * GridBox and layoutOptions options (can be set either in the GridBox itself, or within its child nodes' layoutOptions):
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
 * layoutOptions-only options (can only be set within the child nodes' layoutOptions, NOT available on GridBox):
 *   - x (see https://phetsims.github.io/scenery/doc/layout#GridBox-layoutOptions-location)
 *   - y (see https://phetsims.github.io/scenery/doc/layout#GridBox-layoutOptions-location)
 *   - horizontalSpan (see https://phetsims.github.io/scenery/doc/layout#GridBox-layoutOptions-size)
 *   - verticalSpan (see https://phetsims.github.io/scenery/doc/layout#GridBox-layoutOptions-size)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import assertMutuallyExclusiveOptions from '../../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import optionize from '../../../../phet-core/js/optionize.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import { GRID_CONSTRAINT_OPTION_KEYS, GridCell, GridConstraint, GridConstraintOptions, HorizontalLayoutAlign, LAYOUT_NODE_OPTION_KEYS, LayoutAlign, LayoutNode, LayoutNodeOptions, MarginLayoutCell, Node, REQUIRES_BOUNDS_OPTION_KEYS, scenery, SIZABLE_OPTION_KEYS, VerticalLayoutAlign } from '../../imports.js';

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
  // NOTE: This should be used with the `children` option and/or adding children manually (addChild, etc.)
  // NOTE: This should NOT be used with autoColumns or rows/columns, as those also specify coordinate information
  // NOTE: This will only lay out children with valid bounds, and if excludeInvisibleChildrenFromBounds is true then it
  // will ALSO be constrained to only visible children. It won't leave gaps for children that don't meet these
  // constraints.
  autoRows?: number | null;

  // When non-null, the cells of this grid will be positioned/sized to be 1x1 cells, filling columns until a row has
  // `autoColumns` number of columns, then it will go to the next row. This should generally be used with `children` or
  // adding/removing children in normal ways.
  // NOTE: This should be used with the `children` option and/or adding children manually (addChild, etc.)
  // NOTE: This should NOT be used with autoRows or rows/columns, as those also specify coordinate information
  // NOTE: This will only lay out children with valid bounds, and if excludeInvisibleChildrenFromBounds is true then it
  // will ALSO be constrained to only visible children. It won't leave gaps for children that don't meet these
  // constraints.
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
  private readonly onChildVisibilityToggled: () => void;

  public constructor( providedOptions?: GridBoxOptions ) {
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
    this.onChildVisibilityToggled = this.updateAllAutoLines.bind( this );

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
  public setLines( orientation: Orientation, lineArrays: LineArrays ): void {
    const children: Node[] = [];

    for ( let i = 0; i < lineArrays.length; i++ ) {
      const lineArray = lineArrays[ i ];
      for ( let j = 0; j < lineArray.length; j++ ) {
        const item = lineArray[ j ];
        if ( item !== null ) {
          children.push( item );
          item.mutateLayoutOptions( {
            [ orientation.line ]: i,
            [ orientation.opposite.line ]: j
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
  public getLines( orientation: Orientation ): LineArrays {
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
  public set rows( lineArrays: LineArrays ) {
    this.setLines( Orientation.VERTICAL, lineArrays );
  }

  /**
   * Returns a two-dimensional array of the child Nodes (with null as a spacer) where the inner arrays are the rows.
   */
  public get rows(): LineArrays {
    return this.getLines( Orientation.VERTICAL );
  }

  /**
   * Sets the children of the GridBox by specifying a two-dimensional array of Nodes (or null values as spacers).
   * The inner arrays will be the columns of the grid.
   * * Mutates layoutOptions of the provided Nodes. See setLines() for more documentation.
   */
  public set columns( lineArrays: LineArrays ) {
    this.setLines( Orientation.HORIZONTAL, lineArrays );
  }

  /**
   * Returns a two-dimensional array of the child Nodes (with null as a spacer) where the inner arrays are the columns.
   */
  public get columns(): LineArrays {
    return this.getLines( Orientation.HORIZONTAL );
  }

  /**
   * Returns the Node at a specific row/column intersection (or null if there are none)
   */
  public getNodeAt( row: number, column: number ): Node | null {
    const cell = this.constraint.getCell( row, column );

    return cell ? cell.node : null;
  }

  /**
   * Returns the row index of a child Node (or if it spans multiple rows, the first row)
   */
  public getRowOfNode( node: Node ): number {
    assert && assert( this.children.includes( node ) );

    return this.constraint.getCellFromNode( node )!.position.vertical;
  }

  /**
   * Returns the column index of a child Node (or if it spans multiple columns, the first row)
   */
  public getColumnOfNode( node: Node ): number {
    assert && assert( this.children.includes( node ) );

    return this.constraint.getCellFromNode( node )!.position.horizontal;
  }

  /**
   * Returns all the Nodes in a given row (by index)
   */
  public getNodesInRow( index: number ): Node[] {
    return this.constraint.getCells( Orientation.VERTICAL, index ).map( cell => cell.node );
  }

  /**
   * Returns all the Nodes in a given column (by index)
   */
  public getNodesInColumn( index: number ): Node[] {
    return this.constraint.getCells( Orientation.HORIZONTAL, index ).map( cell => cell.node );
  }

  /**
   * Adds an array of child Nodes (with null allowed as empty spacers) at the bottom of all existing rows.
   */
  public addRow( row: LineArray ): this {

    this.rows = [ ...this.rows, row ];

    return this;
  }

  /**
   * Adds an array of child Nodes (with null allowed as empty spacers) at the right of all existing columns.
   */
  public addColumn( column: LineArray ): this {

    this.columns = [ ...this.columns, column ];

    return this;
  }

  /**
   * Inserts a row of child Nodes at a given row index (see addRow for more information)
   */
  public insertRow( index: number, row: LineArray ): this {

    this.rows = [ ...this.rows.slice( 0, index ), row, ...this.rows.slice( index ) ];

    return this;
  }

  /**
   * Inserts a column of child Nodes at a given column index (see addColumn for more information)
   */
  public insertColumn( index: number, column: LineArray ): this {

    this.columns = [ ...this.columns.slice( 0, index ), column, ...this.columns.slice( index ) ];

    return this;
  }

  /**
   * Removes all child Nodes in a given row
   */
  public removeRow( index: number ): this {

    this.rows = [ ...this.rows.slice( 0, index ), ...this.rows.slice( index + 1 ) ];

    return this;
  }

  /**
   * Removes all child Nodes in a given column
   */
  public removeColumn( index: number ): this {

    this.columns = [ ...this.columns.slice( 0, index ), ...this.columns.slice( index + 1 ) ];

    return this;
  }

  public set autoRows( value: number | null ) {
    assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 1 ) );

    if ( this._autoRows !== value ) {
      this._autoRows = value;

      this.updateAutoRows();
    }
  }

  public get autoRows(): number | null {
    return this._autoRows;
  }

  public set autoColumns( value: number | null ) {
    assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 1 ) );

    if ( this._autoColumns !== value ) {
      this._autoColumns = value;

      this.updateAutoColumns();
    }
  }

  public get autoColumns(): number | null {
    return this._autoColumns;
  }

  // Used for autoRows/autoColumns
  private updateAutoLines( orientation: Orientation, value: number | null ): void {
    if ( value !== null && this._autoLockCount === 0 ) {
      let updatedCount = 0;

      this.constraint.lock();

      this.children.filter( child => {
        return child.bounds.isValid() && ( !this._constraint.excludeInvisible || child.visible );
      } ).forEach( ( child, index ) => {
        const primary = index % value;
        const secondary = Math.floor( index / value );
        const width = 1;
        const height = 1;

        // We guard to see if we actually have to update anything (so we can avoid triggering an auto-layout)
        if ( !child.layoutOptions ||
             child.layoutOptions[ orientation.line ] !== primary ||
             child.layoutOptions[ orientation.opposite.line ] !== secondary ||
             child.layoutOptions.horizontalSpan !== width ||
             child.layoutOptions.verticalSpan !== height
        ) {
          updatedCount++;
          child.mutateLayoutOptions( {
            [ orientation.line ]: index % value,
            [ orientation.opposite.line ]: Math.floor( index / value ),
            horizontalSpan: 1,
            verticalSpan: 1
          } );
        }

      } );

      this.constraint.unlock();

      // Only trigger an automatic layout IF we actually adjusted something.
      if ( updatedCount > 0 ) {
        this.constraint.updateLayoutAutomatically();
      }
    }
  }

  private updateAutoRows(): void {
    this.updateAutoLines( Orientation.VERTICAL, this.autoRows );
  }

  private updateAutoColumns(): void {
    this.updateAutoLines( Orientation.HORIZONTAL, this.autoColumns );
  }

  // Updates rows or columns, whichever is active at the moment (if any)
  private updateAllAutoLines(): void {
    assert && assert( this._autoRows === null || this._autoColumns === null,
      'autoRows and autoColumns should not both be set when updating children' );

    this.updateAutoRows();
    this.updateAutoColumns();
  }

  public override setChildren( children: Node[] ): this {

    const oldChildren = this.getChildren(); // defensive copy

    // Don't update autoRows/autoColumns settings while setting children, wait until after for performance
    this._autoLockCount++;
    super.setChildren( children );
    this._autoLockCount--;

    if ( !_.isEqual( oldChildren, children ) ) {
      this.updateAllAutoLines();
    }

    return this;
  }

  /**
   * Called when a child is inserted.
   */
  private onGridBoxChildInserted( node: Node, index: number ): void {
    node.visibleProperty.lazyLink( this.onChildVisibilityToggled );

    const cell = new GridCell( this._constraint, node, this._constraint.createLayoutProxy( node ) );
    this._cellMap.set( node, cell );

    this._constraint.addCell( cell );

    this.updateAllAutoLines();
  }

  /**
   * Called when a child is removed.
   *
   * NOTE: This is NOT called on disposal. Any additional cleanup (to prevent memory leaks) should be included in the
   * dispose() function
   */
  private onGridBoxChildRemoved( node: Node ): void {

    const cell = this._cellMap.get( node )!;
    assert && assert( cell );

    this._cellMap.delete( node );

    this._constraint.removeCell( cell );

    cell.dispose();

    this.updateAllAutoLines();

    node.visibleProperty.unlink( this.onChildVisibilityToggled );
  }

  public override mutate( options?: GridBoxOptions ): this {
    // children can be used with one of autoRows/autoColumns, but otherwise these options are exclusive
    assertMutuallyExclusiveOptions( options, [ 'rows' ], [ 'columns' ], [ 'children', 'autoRows', 'autoColumns' ] );
    if ( options ) {
      assert && assert( typeof options.autoRows !== 'number' || typeof options.autoColumns !== 'number',
        'autoRows and autoColumns should not be specified both as non-null at the same time' );
    }

    return super.mutate( options );
  }

  public get spacing(): number | number[] {
    return this._constraint.spacing;
  }

  public set spacing( value: number | number[] ) {
    this._constraint.spacing = value;
  }

  public get xSpacing(): number | number[] {
    return this._constraint.xSpacing;
  }

  public set xSpacing( value: number | number[] ) {
    this._constraint.xSpacing = value;
  }

  public get ySpacing(): number | number[] {
    return this._constraint.ySpacing;
  }

  public set ySpacing( value: number | number[] ) {
    this._constraint.ySpacing = value;
  }

  public get xAlign(): HorizontalLayoutAlign {
    return this._constraint.xAlign!;
  }

  public set xAlign( value: HorizontalLayoutAlign ) {
    this._constraint.xAlign = value;
  }

  public get yAlign(): VerticalLayoutAlign {
    return this._constraint.yAlign!;
  }

  public set yAlign( value: VerticalLayoutAlign ) {
    this._constraint.yAlign = value;
  }

  public get grow(): number {
    return this._constraint.grow!;
  }

  public set grow( value: number ) {
    this._constraint.grow = value;
  }

  public get xGrow(): number {
    return this._constraint.xGrow!;
  }

  public set xGrow( value: number ) {
    this._constraint.xGrow = value;
  }

  public get yGrow(): number {
    return this._constraint.yGrow!;
  }

  public set yGrow( value: number ) {
    this._constraint.yGrow = value;
  }

  public get stretch(): boolean {
    return this._constraint.stretch!;
  }

  public set stretch( value: boolean ) {
    this._constraint.stretch = value;
  }

  public get xStretch(): boolean {
    return this._constraint.xStretch!;
  }

  public set xStretch( value: boolean ) {
    this._constraint.xStretch = value;
  }

  public get yStretch(): boolean {
    return this._constraint.yStretch!;
  }

  public set yStretch( value: boolean ) {
    this._constraint.yStretch = value;
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


  public override setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds: boolean ): void {
    super.setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds );

    this.updateAllAutoLines();
  }

  public override dispose(): void {

    // Lock our layout forever
    this._constraint.lock();

    this.childInsertedEmitter.removeListener( this.onChildInserted );
    this.childRemovedEmitter.removeListener( this.onChildRemoved );

    // Dispose our cells here. We won't be getting the children-removed listeners fired (we removed them above)
    for ( const cell of this._cellMap.values() ) {
      cell.dispose();

      cell.node.visibleProperty.unlink( this.onChildVisibilityToggled );
    }

    super.dispose();
  }

  public getHelperNode(): Node {
    const marginsNode = MarginLayoutCell.createHelperNode( this.constraint.displayedCells, this.constraint.layoutBoundsProperty.value, cell => {
      let str = '';

      str += `row: ${cell.position.vertical}\n`;
      str += `column: ${cell.position.horizontal}\n`;
      if ( cell.size.horizontal > 1 ) {
        str += `horizontalSpan: ${cell.size.horizontal}\n`;
      }
      if ( cell.size.vertical > 1 ) {
        str += `verticalSpan: ${cell.size.vertical}\n`;
      }
      str += `xAlign: ${LayoutAlign.internalToAlign( Orientation.HORIZONTAL, cell.effectiveXAlign )}\n`;
      str += `yAlign: ${LayoutAlign.internalToAlign( Orientation.VERTICAL, cell.effectiveYAlign )}\n`;
      str += `xStretch: ${cell.effectiveXStretch}\n`;
      str += `yStretch: ${cell.effectiveYStretch}\n`;
      str += `xGrow: ${cell.effectiveXGrow}\n`;
      str += `yGrow: ${cell.effectiveYGrow}\n`;

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
GridBox.prototype._mutatorKeys = [ ...SIZABLE_OPTION_KEYS, ...GRIDBOX_OPTION_KEYS, ...Node.prototype._mutatorKeys ];

scenery.register( 'GridBox', GridBox );
