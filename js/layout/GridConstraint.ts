// Copyright 2021-2022, University of Colorado Boulder

/**
 * Main grid-layout logic. Usually used indirectly through GridBox, but can also be used directly (say, if nodes don't
 * have the same parent, or a GridBox can't be used).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../phet-core/js/OrientationPair.js';
import mutate from '../../../phet-core/js/mutate.js';
import { GRID_CONFIGURABLE_OPTION_KEYS, GridCell, GridConfigurable, GridConfigurableOptions, GridLine, LayoutAlign, Node, NodeLayoutAvailableConstraintOptions, NodeLayoutConstraint, scenery } from '../imports.js';
import IProperty from '../../../axon/js/IProperty.js';

const GRID_CONSTRAINT_OPTION_KEYS = [
  ...GRID_CONFIGURABLE_OPTION_KEYS,
  'excludeInvisible',
  'spacing',
  'xSpacing',
  'ySpacing'
];

type SelfOptions = {
  spacing?: number;
  xSpacing?: number;
  ySpacing?: number;

  preferredWidthProperty?: IProperty<number | null>;
  preferredHeightProperty?: IProperty<number | null>;
  minimumWidthProperty?: IProperty<number | null>;
  minimumHeightProperty?: IProperty<number | null>;
};

export type GridConstraintOptions = SelfOptions & GridConfigurableOptions & NodeLayoutAvailableConstraintOptions;

export default class GridConstraint extends GridConfigurable( NodeLayoutConstraint ) {

  private readonly cells: Set<GridCell> = new Set();

  // scenery-internal
  displayedCells: GridCell[] = [];

  // Looked up by index
  private displayedLines: OrientationPair<Map<number, GridLine>> = new OrientationPair( new Map(), new Map() );

  private _spacing: OrientationPair<number | number[]> = new OrientationPair<number | number[]>( 0, 0 );

  constructor( ancestorNode: Node, providedOptions?: GridConstraintOptions ) {
    assert && assert( ancestorNode instanceof Node );

    super( ancestorNode, providedOptions );

    this.setConfigToBaseDefault();
    this.mutateConfigurable( providedOptions );
    mutate( this, GRID_CONSTRAINT_OPTION_KEYS, providedOptions );

    // Key configuration changes to relayout
    this.changedEmitter.addListener( this._updateLayoutListener );
  }

  protected override layout(): void {
    super.layout();

    assert && assert( _.every( [ ...this.cells ], cell => !cell.node.isDisposed ) );

    const cells = [ ...this.cells ].filter( cell => {
      return cell.isConnected() && cell.proxy.bounds.isValid() && ( !this.excludeInvisible || cell.node.visible );
    } );
    this.displayedCells = cells;

    if ( !cells.length ) {
      this.layoutBoundsProperty.value = Bounds2.NOTHING;

      // Synchronize our displayedLines, if it's used for display
      this.displayedLines.forEach( map => map.clear() );
      return;
    }

    const minimumSizes = new OrientationPair( 0, 0 );
    const preferredSizes = new OrientationPair( this.preferredWidthProperty.value, this.preferredHeightProperty.value );
    const layoutBounds = new Bounds2( 0, 0, 0, 0 );

    // Handle horizontal first, so if we re-wrap we can handle vertical later.
    [ Orientation.HORIZONTAL, Orientation.VERTICAL ].forEach( orientation => {
      const orientedSpacing = this._spacing.get( orientation );

      // // index => GridLine
      const lineMap: Map<number, GridLine> = this.displayedLines.get( orientation );

      // Clear out the lineMap
      lineMap.forEach( line => line.freeToPool() );
      lineMap.clear();

      const lineIndices = _.sortedUniq( _.sortBy( _.flatten( cells.map( cell => cell.getIndices( orientation ) ) ) ) );
      const lines = lineIndices.map( index => {
        const subCells = _.filter( cells, cell => cell.containsIndex( orientation, index ) );

        const grow = Math.max( ...subCells.map( cell => cell.getEffectiveGrow( orientation ) ) );
        const line = GridLine.pool.create( index, subCells, grow );
        lineMap.set( index, line );

        return line;
      } );

      const lineSpacings = typeof orientedSpacing === 'number' ?
                           _.range( 0, lines.length - 1 ).map( () => orientedSpacing ) :
                           orientedSpacing.slice( 0, lines.length - 1 );
      assert && assert( lineSpacings.length === lines.length - 1 );

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

      const minSizeAndSpacing = _.sum( lines.map( line => line.size ) ) + _.sum( lineSpacings );
      minimumSizes.set( orientation, minSizeAndSpacing );
      const size = Math.max( minSizeAndSpacing, preferredSizes.get( orientation ) || 0 );
      let sizeRemaining = size - minSizeAndSpacing;
      let growableLines;
      while ( sizeRemaining > 1e-7 && ( growableLines = lines.filter( line => {
        return line.grow > 0 && line.size < line.max - 1e-7;
      } ) ).length ) {
        const totalGrow = _.sum( growableLines.map( line => line.grow ) );
        const amountToGrow = Math.min(
          Math.min( ...growableLines.map( line => ( line.max - line.size ) / line.grow ) ),
          sizeRemaining / totalGrow
        );

        assert && assert( amountToGrow > 1e-11 );

        growableLines.forEach( line => {
          line.size += amountToGrow * line.grow;
        } );
        sizeRemaining -= amountToGrow * totalGrow;
      }

      // Layout
      const startPosition = lines[ 0 ].hasOrigin() ? lines[ 0 ].minOrigin : 0;
      layoutBounds[ orientation.minCoordinate ] = startPosition;
      layoutBounds[ orientation.maxCoordinate ] = startPosition + size;
      lines.forEach( ( line, arrayIndex ) => {
        line.position = startPosition + _.sum( lines.slice( 0, arrayIndex ).map( line => line.size ) ) + _.sum( lineSpacings.slice( 0, line.index ) );
      } );
      cells.forEach( cell => {
        const cellIndexPosition = cell.position.get( orientation );
        const cellSize = cell.size.get( orientation );
        const cellLines = cell.getIndices( orientation ).map( index => lineMap.get( index )! );
        const firstLine = lineMap.get( cellIndexPosition )!;
        const cellSpacings = lineSpacings.slice( cellIndexPosition, cellIndexPosition + cellSize - 1 );
        const cellAvailableSize = _.sum( cellLines.map( line => line.size ) ) + _.sum( cellSpacings );
        const cellPosition = firstLine.position;

        cell.reposition(
          orientation,
          cellAvailableSize,
          cellPosition,
          cell.getEffectiveStretch( orientation ),
          -firstLine.minOrigin,
          cell.getEffectiveAlign( orientation )
        );

        const cellBounds = cell.getCellBounds();

        cell.lastAvailableBounds[ orientation.minCoordinate ] = cellPosition;
        cell.lastAvailableBounds[ orientation.maxCoordinate ] = cellPosition + cellAvailableSize;
        cell.lastUsedBounds.set( cellBounds );

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

  get spacing(): number | number[] {
    assert && assert( this.xSpacing === this.ySpacing );

    return this.xSpacing;
  }

  set spacing( value: number | number[] ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._spacing.get( Orientation.HORIZONTAL ) !== value || this._spacing.get( Orientation.VERTICAL ) !== value ) {
      this._spacing.set( Orientation.HORIZONTAL, value );
      this._spacing.set( Orientation.VERTICAL, value );

      this.updateLayoutAutomatically();
    }
  }

  get xSpacing(): number | number[] {
    return this._spacing.get( Orientation.HORIZONTAL );
  }

  set xSpacing( value: number | number[] ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._spacing.get( Orientation.HORIZONTAL ) !== value ) {
      this._spacing.set( Orientation.HORIZONTAL, value );

      this.updateLayoutAutomatically();
    }
  }

  get ySpacing(): number | number[] {
    return this._spacing.get( Orientation.VERTICAL );
  }

  set ySpacing( value: number | number[] ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._spacing.get( Orientation.VERTICAL ) !== value ) {
      this._spacing.set( Orientation.VERTICAL, value );

      this.updateLayoutAutomatically();
    }
  }

  addCell( cell: GridCell ): void {
    assert && assert( cell instanceof GridCell );
    assert && assert( !this.cells.has( cell ) );

    this.cells.add( cell );
    this.addNode( cell.node );
    cell.changedEmitter.addListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  removeCell( cell: GridCell ): void {
    assert && assert( cell instanceof GridCell );
    assert && assert( this.cells.has( cell ) );

    this.cells.delete( cell );
    this.removeNode( cell.node );
    cell.changedEmitter.removeListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  /**
   * Releases references
   */
  override dispose(): void {
    // Lock during disposal to avoid layout calls
    this.lock();

    [ ...this.cells ].forEach( cell => this.removeCell( cell ) );

    super.dispose();

    this.unlock();
  }

  getIndices( orientation: Orientation ): number[] {
    const result: number[] = [];

    this.cells.forEach( cell => {
      result.push( ...cell.getIndices( orientation ) );
    } );

    return _.sortedUniq( _.sortBy( result ) );
  }

  getCell( row: number, column: number ): GridCell | null {
    return _.find( [ ...this.cells ], cell => cell.containsRow( row ) && cell.containsColumn( column ) ) || null;
  }

  getCells( orientation: Orientation, index: number ): GridCell[] {
    return _.filter( [ ...this.cells ], cell => cell.containsIndex( orientation, index ) );
  }

  static create( ancestorNode: Node, options?: GridConstraintOptions ): GridConstraint {
    return new GridConstraint( ancestorNode, options );
  }
}

scenery.register( 'GridConstraint', GridConstraint );
export { GRID_CONSTRAINT_OPTION_KEYS };
