// Copyright 2021-2022, University of Colorado Boulder

/**
 * Displays a background for a given GridConstraint.
 *
 * NOTE: If there are "holes" in the GridBox/GridConstraint (where there is no cell content for an x/y position), then
 * there will be no background for where those cells (if added) would have been.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import assertMutuallyExclusiveOptions from '../../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import optionize from '../../../../phet-core/js/optionize.js';
import { GridCell, GridConstraint, Node, NodeOptions, Rectangle, scenery, TPaint } from '../../imports.js';

type CreateCellBackground = ( gridCell: GridCell ) => Node | null;
type SelfOptions = {
  // Allows full customization of the background for each cell. The cell is passed in, and can be used in any way to
  // generate the background. `cell.lastAvailableBounds` is the bounds to provide. `cell.position.horizontal` and
  // `cell.position.vertical` are the row and column indices of the cell. `cell.size` can also be used.
  createCellBackground?: CreateCellBackground;

  // If no createCellBackground is provided, these will be used for the fill/stroke of the Rectangle created for the
  // cells.
  fill?: TPaint;
  stroke?: TPaint;
};

export type GridBackgroundNodeOptions = SelfOptions & NodeOptions;

export default class GridBackgroundNode extends Node {

  private readonly constraint: GridConstraint;
  private readonly createCellBackground: CreateCellBackground;
  private readonly layoutListener: () => void;

  public constructor( constraint: GridConstraint, providedOptions?: GridBackgroundNodeOptions ) {

    // Don't permit fill/stroke when createCellBackground is provided
    assertMutuallyExclusiveOptions( providedOptions, [ 'createCellBackground' ], [ 'fill', 'stroke' ] );

    const defaultCreateCellBackground = ( cell: GridCell ): Rectangle => {
      return Rectangle.bounds( cell.lastAvailableBounds, {
        fill: options.fill,
        stroke: options.stroke
      } );
    };

    const options = optionize<GridBackgroundNodeOptions, SelfOptions, NodeOptions>()( {
      fill: 'white',
      stroke: 'black',
      createCellBackground: defaultCreateCellBackground
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
  public override dispose(): void {
    this.constraint.finishedLayoutEmitter.removeListener( this.layoutListener );

    super.dispose();
  }
}

scenery.register( 'GridBackgroundNode', GridBackgroundNode );
