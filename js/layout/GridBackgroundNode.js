// Copyright 2021, University of Colorado Boulder

/**
 * Displays a background for a given GridConstraint
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import { scenery, Node, Rectangle, GridConstraint } from '../imports.js';

class GridBackgroundNode extends Node {
  /**
   * @param {GridConstraint} constraint
   * @param {Object} [options]
   */
  constructor( constraint, options ) {
    assert && assert( constraint instanceof GridConstraint );

    options = merge( {
      // {function(GridCell):Node|null}
      createCellBackground: cell => {
        return Rectangle.bounds( cell.lastAvailableBounds, {
          fill: 'white',
          stroke: 'black'
        } );
      }
    }, options );

    super();

    // @private {GridConstraint}
    this.constraint = constraint;

    // @private {function(GridCell):Node|null}
    this.createCellBackground = options.createCellBackground;

    // @private {function}
    this.layoutListener = this.update.bind( this );
    this.constraint.finishedLayoutEmitter.addListener( this.layoutListener );
    this.update();

    this.mutate( options );
  }

  /**
   * @private
   */
  update() {
    this.children = this.constraint.displayedCells.map( this.createCellBackground ).filter( _.identity );
  }

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    this.constraint.finishedLayoutEmitter.removeListener( this.layoutListener );

    super.dispose();
  }
}

scenery.register( 'GridBackgroundNode', GridBackgroundNode );
export default GridBackgroundNode;