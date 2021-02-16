// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import Constraint from './Constraint.js';

class GridConstraint extends Constraint {
  /**
   * @param {Node} rootNode
   */
  constructor( rootNode ) {
    super( rootNode );
  }
}

scenery.register( 'GridConstraint', GridConstraint );

export default GridConstraint;