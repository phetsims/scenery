// Copyright 2021-2022, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../phet-core/js/OrientationPair.js';
import merge from '../../../phet-core/js/merge.js';
import mutate from '../../../phet-core/js/mutate.js';
import { scenery, Node, GridCell, GridConfigurable, GridLine, LayoutConstraint, GRID_CONFIGURABLE_OPTION_KEYS, GridConfigurableOptions } from '../imports.js';
import IProperty from '../../../axon/js/IProperty.js';
import { GridConfigurableAlign } from './GridConfigurable.js';

const GRID_CONSTRAINT_OPTION_KEYS = [
  'excludeInvisible',
  'spacing',
  'xSpacing',
  'ySpacing'
].concat( GRID_CONFIGURABLE_OPTION_KEYS );

type SelfOptions = {
  excludeInvisible?: boolean;
  spacing?: number;
  xSpacing?: number;
  ySpacing?: number;

  preferredWidthProperty?: IProperty<number | null>;
  preferredHeightProperty?: IProperty<number | null>;
  minimumWidthProperty?: IProperty<number | null>;
  minimumHeightProperty?: IProperty<number | null>;
};

export type GridConstraintOptions = SelfOptions & GridConfigurableOptions;

export default class GridConstraint extends GridConfigurable( LayoutConstraint ) {

  private cells: Set<GridCell>;

  // scenery-internal
  displayedCells: GridCell[];

  // Looked up by index
  private displayedLines: OrientationPair<Map<number, GridLine>>;

  private _excludeInvisible: boolean;
  private _spacing: OrientationPair<number | number[]>;

  // Reports out the used layout bounds (may be larger than actual bounds, since it will include margins, etc.)
  layoutBoundsProperty: IProperty<Bounds2>;

  preferredWidthProperty: IProperty<number | null>;
  preferredHeightProperty: IProperty<number | null>;
  minimumWidthProperty: IProperty<number | null>;
  minimumHeightProperty: IProperty<number | null>;

  constructor( ancestorNode: Node, providedOptions?: GridConstraintOptions ) {
    assert && assert( ancestorNode instanceof Node );

    const options = merge( {
      // As options, so we could hook into a Node's preferred/minimum sizes if desired
      preferredWidthProperty: new TinyProperty<number | null>( null ),
      preferredHeightProperty: new TinyProperty<number | null>( null ),
      minimumWidthProperty: new TinyProperty<number | null>( null ),
      minimumHeightProperty: new TinyProperty<number | null>( null )
    }, providedOptions );

    super( ancestorNode );

    this.cells = new Set();
    this.displayedCells = [];
    this.displayedLines = new OrientationPair( new Map(), new Map() );
    this._excludeInvisible = true;
    this._spacing = new OrientationPair<number | number[]>( 0, 0 );

    this.layoutBoundsProperty = new Property( Bounds2.NOTHING, {
      useDeepEquality: true
    } );

    this.preferredWidthProperty = options.preferredWidthProperty;
    this.preferredHeightProperty = options.preferredHeightProperty;
    this.minimumWidthProperty = options.minimumWidthProperty;
    this.minimumHeightProperty = options.minimumHeightProperty;

    this.setConfigToBaseDefault();
    this.mutateConfigurable( options );
    mutate( this, GRID_CONSTRAINT_OPTION_KEYS, options );

    // Key configuration changes to relayout
    this.changedEmitter.addListener( this._updateLayoutListener );

    this.preferredWidthProperty.lazyLink( this._updateLayoutListener );
    this.preferredHeightProperty.lazyLink( this._updateLayoutListener );
  }

  protected override layout() {
    super.layout();

    assert && assert( _.every( [ ...this.cells ], cell => !cell.node.isDisposed ) );

    const cells = [ ...this.cells ].filter( cell => {
      // TODO: Also don't lay out disconnected nodes!!!!
      return cell.node.bounds.isValid() && ( !this._excludeInvisible || cell.node.visible );
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
    const layoutBounds = new Bounds2( 0, 0, 0, 0 ); // TODO: Bounds2.NOTHING.copy() once we have both dimensions handled

    // Handle horizontal first, so if we re-wrap we can handle vertical later.
    [ Orientation.HORIZONTAL, Orientation.VERTICAL ].forEach( orientation => {
      const orientedSpacing = this._spacing.get( orientation );
      const minField = orientation === Orientation.HORIZONTAL ? 'minX' as const : 'minY' as const;
      const maxField = orientation === Orientation.HORIZONTAL ? 'maxX' as const : 'maxY' as const;

      // {Map.<index:number,GridLine>
      const lineMap = this.displayedLines.get( orientation );

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
        }
      } );

      // Then increase for spanning cells as necessary
      cells.forEach( cell => {
        if ( cell.size.get( orientation ) > 1 ) {
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
        line.size = line.min;
      } );
      const minSizeAndSpacing = _.sum( lines.map( line => line.min ) ) + _.sum( lineSpacings );
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
      layoutBounds[ minField ] = 0;
      layoutBounds[ maxField ] = size;
      lines.forEach( ( line, arrayIndex ) => {
        line.position = _.sum( lines.slice( 0, arrayIndex ).map( line => line.size ) ) + _.sum( lineSpacings.slice( 0, line.index ) );
      } );
      cells.forEach( cell => {
        const cellIndexPosition = cell.position.get( orientation );
        const cellSize = cell.size.get( orientation );
        const cellLines = cell.getIndices( orientation ).map( index => lineMap.get( index )! );
        const firstColumn = lineMap.get( cellIndexPosition )!;
        const cellSpacings = lineSpacings.slice( cellIndexPosition, cellIndexPosition + cellSize - 1 );
        const cellAvailableSize = _.sum( cellLines.map( line => line.size ) ) + _.sum( cellSpacings );
        const cellMinimumSize = cell.getMinimumSize( orientation );
        const cellPosition = firstColumn.position;

        const align = cell.getEffectiveAlign( orientation );

        if ( align === GridConfigurableAlign.STRETCH ) {
          cell.attemptPreferredSize( orientation, cellAvailableSize );
        }
        else {
          cell.attemptPreferredSize( orientation, cellMinimumSize );
        }
        cell.attemptPosition( orientation, align, cellPosition, cellAvailableSize );

        const cellBounds = cell.getCellBounds();
        assert && assert( cellBounds.isFinite() );

        cell.lastAvailableBounds[ orientation.minCoordinate ] = cellPosition;
        cell.lastAvailableBounds[ orientation.maxCoordinate ] = cellPosition + cellAvailableSize;
        cell.lastUsedBounds.set( cellBounds );

        layoutBounds[ minField ] = Math.min( layoutBounds[ minField ], cellBounds[ minField ] );
        layoutBounds[ maxField ] = Math.max( layoutBounds[ maxField ], cellBounds[ maxField ] );
      } );
    } );

    // We're taking up these layout bounds (nodes could use them for localBounds)
    this.layoutBoundsProperty.value = layoutBounds;

    this.minimumWidthProperty.value = minimumSizes.horizontal;
    this.minimumHeightProperty.value = minimumSizes.vertical;

    this.finishedLayoutEmitter.emit();
  }

  get excludeInvisible(): boolean {
    return this._excludeInvisible;
  }

  set excludeInvisible( value: boolean ) {
    assert && assert( typeof value === 'boolean' );

    if ( this._excludeInvisible !== value ) {
      this._excludeInvisible = value;

      this.updateLayoutAutomatically();
    }
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

  addCell( cell: GridCell ) {
    assert && assert( cell instanceof GridCell );
    assert && assert( !this.cells.has( cell ) );

    this.cells.add( cell );
    this.addNode( cell.node );
    cell.changedEmitter.addListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  removeCell( cell: GridCell ) {
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
  override dispose() {
    // In case they're from external sources
    this.preferredWidthProperty.unlink( this._updateLayoutListener );
    this.preferredHeightProperty.unlink( this._updateLayoutListener );

    [ ...this.cells ].forEach( cell => this.removeCell( cell ) );

    super.dispose();
  }

  getIndices( orientation: Orientation ): number[] {
    const result: number[] = [];

    this.cells.forEach( cell => {
      result.push( ...cell.getIndices( orientation ) );
    } );

    return _.sortedUniq( _.sortBy( result ) );
  }

  getCell( row: number, column: number ): GridCell | null {
    // TODO: If we have to do ridiculousness like this, just go back to array?
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
