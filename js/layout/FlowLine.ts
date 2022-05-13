// Copyright 2021-2022, University of Colorado Boulder

/**
 * A poolable internal representation of a line for layout handling in FlowConstraint
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Orientation from '../../../phet-core/js/Orientation.js';
import Pool from '../../../phet-core/js/Pool.js';
import { FlowCell, scenery } from '../imports.js';

export default class FlowLine {

  orientation!: Orientation;
  cells!: FlowCell[];
  min!: number;
  max!: number;
  minOrigin!: number;
  maxOrigin!: number;
  size!: number;
  position!: number;

  constructor( orientation: Orientation, cells: FlowCell[] ) {
    this.initialize( orientation, cells );
  }

  initialize( orientation: Orientation, cells: FlowCell[] ): void {

    this.orientation = orientation;
    this.cells = cells;

    this.min = 0;
    this.max = Number.POSITIVE_INFINITY;
    this.minOrigin = Number.POSITIVE_INFINITY;
    this.maxOrigin = Number.NEGATIVE_INFINITY;
    this.size = 0;
    this.position = 0;
  }

  hasOrigin(): boolean {
    return isFinite( this.minOrigin ) && isFinite( this.maxOrigin );
  }

  getMinimumSize( spacing: number ): number {
    return ( this.cells.length - 1 ) * spacing + _.sum( this.cells.map( cell => cell.getMinimumSize( this.orientation ) ) );
  }

  freeToPool(): void {
    FlowLine.pool.freeToPool( this );
  }

  static readonly pool = new Pool<typeof FlowLine, [Orientation, FlowCell[]]>( FlowLine, {
    defaultArguments: [ Orientation.HORIZONTAL, [] ]
  } );
}

scenery.register( 'FlowLine', FlowLine );
