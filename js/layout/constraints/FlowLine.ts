// Copyright 2022-2023, University of Colorado Boulder

/**
 * A poolable internal representation of a line for layout handling in FlowConstraint
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Orientation from '../../../../phet-core/js/Orientation.js';
import Pool from '../../../../phet-core/js/Pool.js';
import { FlowCell, LayoutLine, scenery } from '../../imports.js';

export default class FlowLine extends LayoutLine {

  // (scenery-internal)
  public orientation!: Orientation;
  public cells!: FlowCell[];

  /**
   * (scenery-internal)
   */
  public constructor( orientation: Orientation, cells: FlowCell[] ) {
    super();

    this.initialize( orientation, cells );
  }

  /**
   * (scenery-internal)
   */
  public initialize( orientation: Orientation, cells: FlowCell[] ): this {

    this.orientation = orientation;
    this.cells = cells;

    this.initializeLayoutLine();

    return this;
  }

  /**
   * (scenery-internal)
   */
  public getMinimumSize( spacing: number ): number {
    return ( this.cells.length - 1 ) * spacing + _.sum( this.cells.map( cell => cell.getMinimumSize( this.orientation ) ) );
  }

  /**
   * (scenery-internal)
   */
  public freeToPool(): void {
    FlowLine.pool.freeToPool( this );
  }

  public clean(): void {
    this.cells.length = 0;
    this.freeToPool();
  }

  /**
   * (scenery-internal)
   */
  public static readonly pool = new Pool( FlowLine, {
    defaultArguments: [ Orientation.HORIZONTAL, [] ]
  } );
}

scenery.register( 'FlowLine', FlowLine );
