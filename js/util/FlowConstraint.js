// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';
import Constraint from './Constraint.js';
import FlowConfigurable from './FlowConfigurable.js';

class FlowConstraint extends FlowConfigurable( Constraint ) {
  /**
   * @param {Node} rootNode
   * @param {Object} [options]
   */
  constructor( rootNode, options ) {
    super( rootNode );

    this.setConfigToBaseDefault();
    this.mutateConfigurable( options );
  }
}

scenery.register( 'FlowConstraint', FlowConstraint );

export default FlowConstraint;