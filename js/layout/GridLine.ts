// Copyright 2021, University of Colorado Boulder

/**
 * A poolable internal representation of a row/column for grid handling in GridConstraint
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../phet-core/js/Poolable.js';
import { GridCell, scenery } from '../imports.js';

class GridLine {

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
    // @public {number}
    this.index = index;

    // @public {Array.<GridCell>}
    this.cells = cells;

    // @public {number}
    this.grow = grow;
    this.min = 0;
    this.max = Number.POSITIVE_INFINITY;
    this.size = 0;
    this.position = 0;
  }
}

// Sets up pooling on GridLine
Poolable.mixInto( GridLine, {
  defaultArguments: [ 0, [], 0 ]
} );

scenery.register( 'GridLine', GridLine );
export default GridLine;