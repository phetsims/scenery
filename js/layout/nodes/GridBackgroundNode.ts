// Copyright 2021-2022, University of Colorado Boulder

/**
 * Displays a background for a given GridConstraint
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import optionize from '../../../../phet-core/js/optionize.js';
import { GridCell, GridConstraint, Node, NodeOptions, Rectangle, scenery } from '../../imports.js';

type CreateCellBackground = ( gridCell: GridCell ) => Node | null;
type SelfOptions = {
  createCellBackground?: CreateCellBackground;
};

export type GridBackgroundNodeOptions = SelfOptions & NodeOptions;

export default class GridBackgroundNode extends Node {

  private readonly constraint: GridConstraint;
  private readonly createCellBackground: CreateCellBackground;
  private readonly layoutListener: () => void;

  constructor( constraint: GridConstraint, providedOptions?: GridBackgroundNodeOptions ) {
    assert && assert( constraint instanceof GridConstraint );

    const options = optionize<GridBackgroundNodeOptions, SelfOptions, NodeOptions>()( {
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
