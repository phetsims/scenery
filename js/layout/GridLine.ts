// Copyright 2021-2022, University of Colorado Boulder

/**
 * A poolable internal representation of a row/column for grid handling in GridConstraint
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Pool from '../../../phet-core/js/Pool.js';
import { GridCell, scenery } from '../imports.js';

export default class GridLine {

  index!: number;
  cells!: GridCell[];
  grow!: number;
  min!: number;
  max!: number;
  size!: number;
  position!: number;

  constructor( index: number, cells: GridCell[], grow: number ) {
    this.initialize( index, cells, grow );
  }

  initialize( index: number, cells: GridCell[], grow: number ) {
    // {number}
    this.index = index;

    // {Array.<GridCell>}
    this.cells = cells;

    // {number}
    this.grow = grow;
    this.min = 0;
    this.max = Number.POSITIVE_INFINITY;
    this.size = 0;
    this.position = 0;
  }

  freeToPool() {
    GridLine.pool.freeToPool( this );
  }

  static pool = new Pool<typeof GridLine, [number, GridCell[], number]>( GridLine, {
    defaultArguments: [ 0, [], 0 ]
  } );
}

scenery.register( 'GridLine', GridLine );
