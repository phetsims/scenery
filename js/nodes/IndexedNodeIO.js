// Copyright 2020, University of Colorado Boulder

/**
 * IO Type for Nodes that can save their own index (if phetioState: true).  Can be used to customize z-order
 * or layout order.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import IOType from '../../../tandem/js/types/IOType.js';
import VoidIO from '../../../tandem/js/types/VoidIO.js';
import scenery from '../scenery.js';
import Node from './Node.js';

const IndexedNodeIO = new IOType( 'IndexedNodeIO', {
  valueType: Node,
  documentation: 'Node that can be moved forward/back by index, which specifies z-order and/or layout order',
  supertype: Node.NodeIO,
  toStateObject: node => {
    const stateObject = {};
    if ( node.parents[ 0 ] ) {
      stateObject.index = node.parents[ 0 ].indexOfChild( node );
    }
    return stateObject;
  },
  applyState: ( node, fromStateObject ) => {
    if ( node.parents[ 0 ] ) {
      node.parents[ 0 ].moveChildToIndex( node, fromStateObject.index );
    }
  },
  methods: {
    moveForward: {
      returnType: VoidIO,
      parameterTypes: [],
      implementation: function() {
        return this.moveForward();
      },
      documentation: 'Move this node one index forward in each of its parents.  If the node is already at the front, this is a no-op.'
    },

    moveBackward: {
      returnType: VoidIO,
      parameterTypes: [],
      implementation: function() {
        return this.moveBackward();
      },
      documentation: 'Move this node one index backward in each of its parents.  If the node is already at the back, this is a no-op.'
    }
  }
} );

scenery.register( 'IndexedNodeIO', IndexedNodeIO );
export default IndexedNodeIO;