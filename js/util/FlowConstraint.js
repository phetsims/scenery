// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import Constraint from './Constraint.js';

class FlowConstraint extends Constraint {
  /**
   * @param {Node} rootNode
   */
  constructor( rootNode ) {
    super( rootNode );
  }
}

scenery.register( 'FlowConstraint', FlowConstraint );

export default FlowConstraint;