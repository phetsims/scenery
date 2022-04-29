// Copyright 2021-2022, University of Colorado Boulder

/**
 * Displays a background for a given GridConstraint
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import { scenery, Node, Rectangle, GridConstraint, GridCell, NodeOptions } from '../imports.js';

type CreateCellBackground = ( gridCell: GridCell ) => Node | null;
type SelfOptions = {
  createCellBackground: CreateCellBackground;
};

export type GridBackgroundNodeOptions = SelfOptions & NodeOptions;

export default class GridBackgroundNode extends Node {

  private constraint: GridConstraint;
  private createCellBackground: CreateCellBackground;
  private layoutListener: () => void;

  constructor( constraint: GridConstraint, providedOptions?: GridBackgroundNodeOptions ) {
    assert && assert( constraint instanceof GridConstraint );

    const options = merge( {
      // {function(GridCell):Node|null}
      createCellBackground: ( cell: GridCell ) => {
        return Rectangle.bounds( cell.lastAvailableBounds, {
          fill: 'white',
          stroke: 'black'
        } );
      }
    }, providedOptions );

    super();

    this.constraint = constraint;
    this.createCellBackground = options.createCellBackground;
    this.layoutListener = this.update.bind( this );
    this.constraint.finishedLayoutEmitter.addListener( this.layoutListener );
    this.update();

    this.mutate( options );
  }

  private update(): void {
    this.children = this.constraint.displayedCells.map( this.createCellBackground ).filter( _.identity ) as Node[];
  }

  /**
   * Releases references
   */
  override dispose(): void {
    this.constraint.finishedLayoutEmitter.removeListener( this.layoutListener );

    super.dispose();
  }
}

scenery.register( 'GridBackgroundNode', GridBackgroundNode );
