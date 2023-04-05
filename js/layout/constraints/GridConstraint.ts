// Copyright 2021-2023, University of Colorado Boulder

/**
 * Main grid-layout logic. Usually used indirectly through GridBox, but can also be used directly (say, if nodes don't
 * have the same parent, or a GridBox can't be used).
 *
 * Throughout the documentation for grid-related items, the term "line" refers to either a row or column (depending on
 * the orientation).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../../dot/js/Bounds2.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../../phet-core/js/OrientationPair.js';
import mutate from '../../../../phet-core/js/mutate.js';
import { ExternalGridConfigurableOptions, GRID_CONFIGURABLE_OPTION_KEYS, GridCell, GridConfigurable, GridLine, LayoutAlign, Node, NodeLayoutAvailableConstraintOptions, NodeLayoutConstraint, scenery } from '../../imports.js';
import TProperty from '../../../../axon/js/TProperty.js';

const GRID_CONSTRAINT_OPTION_KEYS = [
  ...GRID_CONFIGURABLE_OPTION_KEYS,
  'excludeInvisible',
  'spacing',
  'xSpacing',
  'ySpacing'
];

type SelfOptions = {

  // Spacings are controlled in each dimension (setting `spacing`) will adjust both. If it's a number, it will be an
  // extra gap in-between every row or column. If it's an array, it will specify the gap between successive rows/columns
  // e.g. [ 5, 4 ] will have a spacing of 5 between the first and second lines, and 4 between the second and third
  // lines. In that case, if there were a third line, it would have zero spacing between the second (any non-specified
  // spacings for extra rows/columns will be zero).
  // NOTE: If a line (row/column) is invisible (and excludeInvisible is set to true), then the spacing that is directly
  // after (to the right/bottom of) that line will be ignored.
  spacing?: number | number[];
  xSpacing?: number | number[];
  ySpacing?: number | number[];

  // The preferred width/height (ideally from a container's localPreferredWidth/localPreferredHeight.
  preferredWidthProperty?: TProperty<number | null>;
  preferredHeightProperty?: TProperty<number | null>;

  // The minimum width/height (ideally from a container's localMinimumWidth/localMinimumHeight.
  minimumWidthProperty?: TProperty<number | null>;
  minimumHeightProperty?: TProperty<number | null>;
};
type ParentOptions = ExternalGridConfigurableOptions & NodeLayoutAvailableConstraintOptions;
export type GridConstraintOptions = SelfOptions & ParentOptions;

export default class GridConstraint extends GridConfigurable( NodeLayoutConstraint ) {

  private readonly cells = new Set<GridCell>();

  // (scenery-internal)
  public displayedCells: GridCell[] = [];

  // (scenery-internal) Looked up by index
  public displayedLines = new OrientationPair<Map<number, GridLine>>( new Map(), new Map() );

  private _spacing: OrientationPair<number | number[]> = new OrientationPair<number | number[]>( 0, 0 );

  public constructor( ancestorNode: Node, providedOptions?: GridConstraintOptions ) {
    super( ancestorNode, providedOptions );

    // Set configuration to actual default values (instead of null) so that we will have guaranteed non-null
    // (non-inherit) values for our computations.
    this.setConfigToBaseDefault();
    this.mutateConfigurable( providedOptions );
    mutate( this, GRID_CONSTRAINT_OPTION_KEYS, providedOptions );

    // Key configuration changes to relayout
    this.changedEmitter.addListener( this._updateLayoutListener );
  }

  protected override layout(): void {
    super.layout();

    // Only grab the cells that will participate in layout
    const cells = this.filterLayoutCells( [ ...this.cells ] );
    this.displayedCells = cells;

    if ( !cells.length ) {
      this.layoutBoundsProperty.value = Bounds2.NOTHING;
      this.minimumWidthProperty.value = null;
      this.minimumHeightProperty.value = null;

      // Synchronize our displayedLines, if it's used for display (e.g. GridBackgroundNode)
      this.displayedLines.forEach( map => map.clear() );
      return;
    }

    const minimumSizes = new OrientationPair( 0, 0 );
    const preferredSizes = new OrientationPair( this.preferredWidthProperty.value, this.preferredHeightProperty.value );
    const layoutBounds = new Bounds2( 0, 0, 0, 0 );

    // Handle horizontal first, so if we re-wrap we can handle vertical later.
    [ Orientation.HORIZONTAL, Orientation.VERTICAL ].forEach( orientation => {
      const orientedSpacing = this._spacing.get( orientation );

      // index => GridLine
      const lineMap: Map<number, GridLine> = this.displayedLines.get( orientation );

      // Clear out the lineMap
      lineMap.forEach( line => line.clean() );
      lineMap.clear();

      // What are all the line indices used by displayed cells? There could be gaps. We pretend like those gaps
      // don't exist (except for spacing)
      const lineIndices = _.sortedUniq( _.sortBy( _.flatten( cells.map( cell => cell.getIndices( orientation ) ) ) ) );

      const lines = lineIndices.map( index => {
        // Recall, cells can include multiple lines in the same orientation if they have width/height>1
        const subCells = _.filter( cells, cell => cell.containsIndex( orientation, index ) );

        // For now, we'll use the maximum grow value included in this line
        const grow = Math.max( ...subCells.map( cell => cell.getEffectiveGrow( orientation ) ) );

        const line = GridLine.pool.create( index, subCells, grow );
        lineMap.set( index, line );

        return line;
      } );

      // Convert a simple spacing number (or a spacing array) into a spacing array of the correct size, only including
      // spacings AFTER our actually-visible lines. We'll also skip the spacing after the last line, as it won't be used
      const lineSpacings = lines.slice( 0, -1 ).map( line => {
        return typeof orientedSpacing === 'number' ? orientedSpacing : orientedSpacing[ line.index ];
      } );

      // Scan sizes for single-line cells first
      cells.forEach( cell => {
        if ( cell.size.get( orientation ) === 1 ) {
          const line = lineMap.get( cell.position.get( orientation ) )!;
          line.min = Math.max( line.min, cell.getMinimumSize( orientation ) );
          line.max = Math.min( line.max, cell.getMaximumSize( orientation ) );

          // For origin-specified cells, we will record their maximum reach from the origin, so these can be "summed"
          // (since the origin line may end up taking more space).
          if ( cell.getEffectiveAlign( orientation ) === LayoutAlign.ORIGIN ) {
            const originBounds = cell.getOriginBounds();
            line.minOrigin = Math.min( originBounds[ orientation.minCoordinate ], line.minOrigin );
            line.maxOrigin = Math.max( originBounds[ orientation.maxCoordinate ], line.maxOrigin );
          }
        }
      } );

      // Then increase for spanning cells as necessary
      cells.forEach( cell => {
        if ( cell.size.get( orientation ) > 1 ) {
          assert && assert( cell.getEffectiveAlign( orientation ) !== LayoutAlign.ORIGIN, 'origin alignment cannot be specified for cells that span >1 width or height' );
          // TODO: don't bump mins over maxes here (if lines have maxes, redistribute otherwise)
          // TODO: also handle maxes
          const lines = cell.getIndices( orientation ).map( index => lineMap.get( index )! );
          const currentMin = _.sum( lines.map( line => line.min ) );
          const neededMin = cell.getMinimumSize( orientation );
          if ( neededMin > currentMin ) {
            const lineDelta = ( neededMin - currentMin ) / lines.length;
            lines.forEach( line => {
              line.min += lineDelta;
            } );
          }
        }
      } );

      // Adjust line sizes to the min
      lines.forEach( line => {
        // If we have origin-specified content, we'll need to include the maximum origin span (which may be larger)
        if ( line.hasOrigin() ) {
          line.size = Math.max( line.min, line.maxOrigin - line.minOrigin );
        }
        else {
          line.size = line.min;
        }
      } );

      // Minimum size of our grid in this orientation
      const minSizeAndSpacing = _.sum( lines.map( line => line.size ) ) + _.sum( lineSpacings );
      minimumSizes.set( orientation, minSizeAndSpacing );

      // Compute the size in this orientation (growing the size proportionally in lines as necessary)
      const size = Math.max( minSizeAndSpacing, preferredSizes.get( orientation ) || 0 );
      let sizeRemaining = size - minSizeAndSpacing;
      let growableLines;
      while ( sizeRemaining > 1e-7 && ( growableLines = lines.filter( line => {
        return line.grow > 0 && line.size < line.max - 1e-7;
      } ) ).length ) {
        const totalGrow = _.sum( growableLines.map( line => line.grow ) );

        // We could need to stop growing EITHER when a line hits its max OR when we run out of space remaining.
        const amountToGrow = Math.min(
          Math.min( ...growableLines.map( line => ( line.max - line.size ) / line.grow ) ),
          sizeRemaining / totalGrow
        );

        assert && assert( amountToGrow > 1e-11 );

        // Grow proportionally to their grow values
        growableLines.forEach( line => {
          line.size += amountToGrow * line.grow;
        } );
        sizeRemaining -= amountToGrow * totalGrow;
      }

      // Layout
      const startPosition = ( lines[ 0 ].hasOrigin() ? lines[ 0 ].minOrigin : 0 ) + this.layoutOriginProperty.value[ orientation.coordinate ];
      layoutBounds[ orientation.minCoordinate ] = startPosition;
      layoutBounds[ orientation.maxCoordinate ] = startPosition + size;
      lines.forEach( ( line, arrayIndex ) => {
        // Position all the lines
        const totalPreviousLineSizes = _.sum( lines.slice( 0, arrayIndex ).map( line => line.size ) );
        const totalPreviousSpacings = _.sum( lineSpacings.slice( 0, arrayIndex ) );
        line.position = startPosition + totalPreviousLineSizes + totalPreviousSpacings;
      } );
      cells.forEach( cell => {
        // The line index of the first line our cell is composed of.
        const cellFirstIndexPosition = cell.position.get( orientation );

        // The size of our cell (width/height)
        const cellSize = cell.size.get( orientation );

        // The line index of the last line our cell is composed of.
        const cellLastIndexPosition = cellFirstIndexPosition + cellSize - 1;

        // All the lines our cell is composed of.
        const cellLines = cell.getIndices( orientation ).map( index => lineMap.get( index )! );

        const firstLine = lineMap.get( cellFirstIndexPosition )!;

        // If we're spanning multiple lines, we have to include the spacing that we've "absorbed" (if we have a cell
        // that spans columns 2 and 3, we'll need to include the spacing between 2 and 3.
        let interiorAbsorbedSpacing = 0;
        if ( cellFirstIndexPosition !== cellLastIndexPosition ) {
          lines.slice( 0, -1 ).forEach( ( line, lineIndex ) => {
            if ( line.index >= cellFirstIndexPosition && line.index < cellLastIndexPosition ) {
              interiorAbsorbedSpacing += lineSpacings[ lineIndex ];
            }
          } );
        }

        // Our size includes the line sizes and spacings
        const cellAvailableSize = _.sum( cellLines.map( line => line.size ) ) + interiorAbsorbedSpacing;
        const cellPosition = firstLine.position;

        // Adjust preferred size and move the cell
        const cellBounds = cell.reposition(
          orientation,
          cellAvailableSize,
          cellPosition,
          cell.getEffectiveStretch( orientation ),
          -firstLine.minOrigin,
          cell.getEffectiveAlign( orientation )
        );

        layoutBounds[ orientation.minCoordinate ] = Math.min( layoutBounds[ orientation.minCoordinate ], cellBounds[ orientation.minCoordinate ] );
        layoutBounds[ orientation.maxCoordinate ] = Math.max( layoutBounds[ orientation.maxCoordinate ], cellBounds[ orientation.maxCoordinate ] );
      } );
    } );

    // We're taking up these layout bounds (nodes could use them for localBounds)
    this.layoutBoundsProperty.value = layoutBounds;

    this.minimumWidthProperty.value = minimumSizes.horizontal;
    this.minimumHeightProperty.value = minimumSizes.vertical;

    this.finishedLayoutEmitter.emit();
  }

  public get spacing(): number | number[] {
    assert && assert( this.xSpacing === this.ySpacing );

    return this.xSpacing;
  }

  public set spacing( value: number | number[] ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._spacing.get( Orientation.HORIZONTAL ) !== value || this._spacing.get( Orientation.VERTICAL ) !== value ) {
      this._spacing.set( Orientation.HORIZONTAL, value );
      this._spacing.set( Orientation.VERTICAL, value );

      this.updateLayoutAutomatically();
    }
  }

  public get xSpacing(): number | number[] {
    return this._spacing.get( Orientation.HORIZONTAL );
  }

  public set xSpacing( value: number | number[] ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._spacing.get( Orientation.HORIZONTAL ) !== value ) {
      this._spacing.set( Orientation.HORIZONTAL, value );

      this.updateLayoutAutomatically();
    }
  }

  public get ySpacing(): number | number[] {
    return this._spacing.get( Orientation.VERTICAL );
  }

  public set ySpacing( value: number | number[] ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._spacing.get( Orientation.VERTICAL ) !== value ) {
      this._spacing.set( Orientation.VERTICAL, value );

      this.updateLayoutAutomatically();
    }
  }

  public addCell( cell: GridCell ): void {
    assert && assert( !this.cells.has( cell ) );

    this.cells.add( cell );
    this.addNode( cell.node );
    cell.changedEmitter.addListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  public removeCell( cell: GridCell ): void {
    assert && assert( this.cells.has( cell ) );

    this.cells.delete( cell );
    this.removeNode( cell.node );
    cell.changedEmitter.removeListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  /**
   * Releases references
   */
  public override dispose(): void {
    // Lock during disposal to avoid layout calls
    this.lock();

    [ ...this.cells ].forEach( cell => this.removeCell( cell ) );

    this.displayedLines.forEach( map => map.clear() );
    this.displayedCells = [];

    super.dispose();

    this.unlock();
  }

  public getIndices( orientation: Orientation ): number[] {
    const result: number[] = [];

    this.cells.forEach( cell => {
      result.push( ...cell.getIndices( orientation ) );
    } );

    return _.sortedUniq( _.sortBy( result ) );
  }

  public getCell( row: number, column: number ): GridCell | null {
    return _.find( [ ...this.cells ], cell => cell.containsRow( row ) && cell.containsColumn( column ) ) || null;
  }

  public getCellFromNode( node: Node ): GridCell | null {
    return _.find( [ ...this.cells ], cell => cell.node === node ) || null;
  }

  public getCells( orientation: Orientation, index: number ): GridCell[] {
    return _.filter( [ ...this.cells ], cell => cell.containsIndex( orientation, index ) );
  }

  public static create( ancestorNode: Node, options?: GridConstraintOptions ): GridConstraint {
    return new GridConstraint( ancestorNode, options );
  }
}

scenery.register( 'GridConstraint', GridConstraint );
export { GRID_CONSTRAINT_OPTION_KEYS };
