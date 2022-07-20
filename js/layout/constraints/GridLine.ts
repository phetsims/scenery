// Copyright 2021-2022, University of Colorado Boulder

/**
 * A poolable internal representation of a row/column for grid handling in GridConstraint
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Pool from '../../../../phet-core/js/Pool.js';
import { GridCell, LayoutLine, scenery } from '../../imports.js';

export default class GridLine extends LayoutLine {

  // (scenery-internal)
  public index!: number;
  public cells!: GridCell[];
  public grow!: number;

  /**
   * (scenery-internal)
   */
  public constructor( index: number, cells: GridCell[], grow: number ) {
    super();

    this.initialize( index, cells, grow );
  }

  /**
   * (scenery-internal)
   */
  public initialize( index: number, cells: GridCell[], grow: number ): void {
    this.index = index;

    this.cells = cells;

    this.grow = grow;

    this.initializeLayoutLine();
  }

  /**
   * (scenery-internal)
   */
  public freeToPool(): void {
    GridLine.pool.freeToPool( this );
  }

  /**
   * (scenery-internal)
   */
  public static readonly pool = new Pool<typeof GridLine, [number, GridCell[], number]>( GridLine, {
    defaultArguments: [ 0, [], 0 ]
  } );
}

scenery.register( 'GridLine', GridLine );
