// Copyright 2021-2025, University of Colorado Boulder

/**
 * Main grid-layout logic. Usually used indirectly through GridBox, but can also be used directly (say, if nodes don't
 * have the same parent, or a GridBox can't be used).
 *
 * Throughout the documentation for grid-related items, the term "line" refers to either a row or column (depending on
 * the orientation).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TProperty from '../../../../axon/js/TProperty.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import mutate from '../../../../phet-core/js/mutate.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../../phet-core/js/OrientationPair.js';
import type { ExternalGridConfigurableOptions } from '../../layout/constraints/GridConfigurable.js';
import { GRID_CONFIGURABLE_OPTION_KEYS } from '../../layout/constraints/GridConfigurable.js';
import GridCell from '../../layout/constraints/GridCell.js';
import GridConfigurable from '../../layout/constraints/GridConfigurable.js';
import GridLine from '../../layout/constraints/GridLine.js';
import LayoutAlign from '../../layout/LayoutAlign.js';
import Node from '../../nodes/Node.js';
import { NodeLayoutAvailableConstraintOptions, default as NodeLayoutConstraint } from '../../layout/constraints/NodeLayoutConstraint.js';
import scenery from '../../scenery.js';

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

      const cellToLinesMap = new Map<GridCell, GridLine[]>( cells.map( cell => {
        return [ cell, cell.getIndices( orientation ).map( index => {
          const line = lineMap.get( index )!;
          assert && assert( line );

          return line;
        } ) ];
      } ) );
      const linesIn = ( cell: GridCell ) => cellToLinesMap.get( cell )!;

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
      {
        // Problem:
        //   Cells that span multiple lines (e.g. horizontalSpan/verticalSpan > 1) may be larger than the sum of their
        //   lines' sizes. We need to grow the lines to accommodate these cells.
        //
        // Constraint:
        //   Cells also can have maximum sizes. We don't want to grow lines in a way that would break this type of constraint.
        //
        // Goals:
        //   Do the above, but try to spread out extra space proportionally. If `grow` is specified on cells (a line),
        //   try to use that to dump space into those sections.

        // An iterative approach, where we will try to grow lines.
        //
        // Every step, we will determine:
        // 1. Grow constraints (how much is the most we'd need to grow things to satisfy an unsatisfied cell)
        // 2. Max constraints (some cells might have a maximum size constraint)
        // 3. Growable lines (the lines, given the above, that have a non-zero amount they can grow).
        // 4. Weights for each growable line (HOW FAST we should grow the lines, proportionally).
        //
        // We will then see how much we can grow before hitting the FIRST constraint, do that, and then continue
        // iteration.
        //
        // This is complicated by the fact that some "max constraints" might have grown enough that none of their
        // lines should increase in size. When that happens, we'll need to add those lines to the "forbidden" list,
        // and recompute things.

        // NOTE: If this is suboptimal, we could try a force-directed iterative algorithm to minimize a specific loss
        // function (or just... convex optimization).

        const epsilon = 1e-9;

        // What is the current (minimum) size of a cell
        const getCurrentCellSize = ( cell: GridCell ) => {
          return _.sum( linesIn( cell ).map( line => line.min ) );
        };

        // A cell is "unsatisfied" if its current size is less than its minimum size
        const isUnsatisfied = ( cell: GridCell ) => {
          return getCurrentCellSize( cell ) < cell.getMinimumSize( orientation ) - epsilon;
        };

        // We may need to "forbid" certain lines from growing further, even if they are not at THEIR limit.
        // Then we can recompute things based on that lines are allowed to grow.
        const forbiddenLines = new Set<GridLine>();

        // Whether our line can grow (i.e. it's not at its maximum size, and it's not forbidden)
        const lineCanGrow = ( line: GridLine ) => {
          return ( !isFinite( line.max ) || line.min < line.max - epsilon ) && !forbiddenLines.has( line );
        };

        // Cells that:
        // 1. Span multiple lines (since we handled single-line cells above)
        // 2. Are unsatisfied (i.e. their current size is less than their minimum size)
        let unsatisfiedSpanningCells = cells.filter( cell => cell.size.get( orientation ) > 1 ).filter( isUnsatisfied );

        // NOTE: It may be possible to actually SKIP the above "single-line cell" step, and size things up JUST using
        // this algorithm. It is unclear whether it would result in decent quality.

        // We'll iterate until we have satisfied all of the unsatisfied cells OR we'll bail (with a break) if
        // we can't grow any further. Otherwise, every step will grow at least something, so we will make incremental
        // forward progress.
        while ( unsatisfiedSpanningCells.length ) {

          // A Grow or Max constraint
          type Constraint = { cell: GridCell; growingLines: GridLine[]; space: number };

          // Specializations for Grow constraints
          type GrowConstraint = Constraint & { weights: Map<GridLine, number> };

          // Gets the applicable (non-zero) grow constraint for a cell, or returns null if it does not need to be grown
          // (or cannot be grown and doesn't fail assertions).
          // We are somewhat permissive here for the runtime, even if assertions would fail we'll try to keep going.
          const getGrowConstraint = ( cell: GridCell ): GrowConstraint | null => {
            assert && assert( isUnsatisfied( cell ) );

            const growableLines = linesIn( cell ).filter( lineCanGrow );

            assert && assert( growableLines.length, 'GridCell for Node cannot find space due to maximum-size constraints' );
            if ( !growableLines.length ) {
              return null;
            }

            const currentSize = getCurrentCellSize( cell );
            const neededMinSize = cell.getMinimumSize( orientation );

            // How much space will need to be added (in total) to satisfy the cell.
            const space = neededMinSize - currentSize;

            assert && assert( space > 0 );
            if ( space < epsilon ) {
              return null;
            }

            // We'll check the "grow" values, IF there are any non-zero. If there are, we will use those proportionally
            // to reallocate space.
            //
            // IF THERE IS NO NON-ZERO GROW VALUES, we will just evenly distribute the space.
            const totalLinesGrow = _.sum( growableLines.map( line => line.grow ) );

            return {
              cell: cell,
              growingLines: growableLines.slice(), // defensive copy, we may modify these later
              space: space,
              weights: new Map( growableLines.map( line => {
                let weight: number;

                if ( totalLinesGrow > 0 ) {
                  weight = space * ( line.grow / totalLinesGrow );
                }
                else {
                  weight = space / growableLines.length;
                }

                return [ line, weight ];
              } ) )
            };
          };

          // Initialize grow constraints
          let growConstraints: GrowConstraint[] = [];
          for ( const cell of unsatisfiedSpanningCells ) {
            const growConstraint = getGrowConstraint( cell );
            if ( growConstraint ) {
              growConstraints.push( growConstraint );
            }
          }

          // We'll need to recompute. We'll see which ones we can keep and which we can recompute (based on forbidden lines)
          const recomputeGrowConstraints = () => {
            growConstraints = growConstraints.map( constraint => {
              if ( constraint.growingLines.some( line => forbiddenLines.has( line ) ) ) {
                return getGrowConstraint( constraint.cell );
              }
              else {
                return constraint;
              }
            } ).filter( constraint => constraint !== null );
          };

          // Find all of the necessary max constraints that are relevant
          let maxConstraints: Constraint[] = [];
          let changed = true;
          while ( changed && growConstraints.length ) {
            // We'll need to iterate, and recompute constraints if our grow constraints have changed
            changed = false;
            maxConstraints = [];

            // Find cells that can't grow any further. They are either:
            // 1. Already at their maximum size - we may need to add forbidden lines
            // 2. Not at their maximum size - we might add them as constraints
            for ( const cell of cells ) {
              const max = cell.getMaximumSize( orientation );

              // Most cells will probably have an infinite max (e.g. NO maximum), so we'll skip those
              if ( isFinite( max ) ) {
                // Find out which lines are "dynamic" (i.e. they are part of a grow constraint)
                // eslint-disable-next-line @typescript-eslint/no-loop-func
                const dynamicLines = linesIn( cell ).filter( line => growConstraints.some( constraint => constraint.growingLines.includes( line ) ) );

                // If there are none, it is not relevant, and we can skip it (won't affect any lines we are considering growing)
                if ( dynamicLines.length ) {
                  const currentSize = getCurrentCellSize( cell );

                  const space = max - currentSize;

                  // Check any "ungrowable" constraints, and remove all of their lines from consideration.
                  if ( space < epsilon ) {
                    for ( const badLine of dynamicLines ) {
                      assert && assert( !forbiddenLines.has( badLine ), 'New only' );

                      forbiddenLines.add( badLine );
                    }

                    changed = true;
                  }
                  else {
                    // If we have space, we'll want to mark how much space
                    maxConstraints.push( { cell: cell, growingLines: dynamicLines.slice(), space: space } );
                  }
                }
              }
            }

            // If ANY forbidden lines changed, recompute grow constraints and try again
            if ( changed ) {
              recomputeGrowConstraints();
            }
          }

          // Actual growing operation
          if ( growConstraints.length ) {
            // Which lines will we increase?
            const growingLines = _.uniq( growConstraints.flatMap( constraint => constraint.growingLines ) );

            // Sum up weights from different constraints
            const weightMap = new Map<GridLine, number>( growingLines.map( line => {
              let weight = 0;

              for ( const constraint of growConstraints ) {
                if ( constraint.growingLines.includes( line ) ) {
                  weight += constraint.weights.get( line )!;
                }
              }

              assert && assert( isFinite( weight ) );

              return [ line, weight ];
            } ) );

            // Find the multiplier that will allow us to (maximally) satisfy all the constraints.
            // Later: increase for each line is multiplier * weight.
            let multiplier = Number.POSITIVE_INFINITY;

            // Minimize the multiplier by what will not violate any constraints
            for ( const constraint of [ ...growConstraints, ...maxConstraints ] ) {
              // Our "total" weight, i.e. how much space we gain if we had a multiplier of 1.
              const velocity = _.sum( constraint.growingLines.map( line => weightMap.get( line )! ) );

              // Adjust the multiplier
              multiplier = Math.min( multiplier, constraint.space / velocity );
              assert && assert( isFinite( multiplier ) && multiplier > 0 );
            }

            // Apply the multiplier
            for ( const line of growingLines ) {
              const velocity = weightMap.get( line )!;
              line.min += velocity * multiplier;
            }

            // Now see which cells are unsatisfied still (so we can iterate)
            unsatisfiedSpanningCells = unsatisfiedSpanningCells.filter( isUnsatisfied );
          }
          else {
            // Bail (so we don't hard error) if we can't grow any further (but are still unsatisfied).
            // This might result from maxContentSize constraints, where it is not possible to expand further.
            assert && assert( false, 'GridCell for Node cannot find space due to maximum-size constraints' );
            break;
          }
        }
      }

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
        const cellLines = linesIn( cell );

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