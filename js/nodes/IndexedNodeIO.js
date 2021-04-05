// Copyright 2020, University of Colorado Boulder

/**
 * IO Type for Nodes that can save their own index (if phetioState: true).  Can be used to customize z-order
 * or layout order.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import FunctionIO from '../../../tandem/js/types/FunctionIO.js';
import IOType from '../../../tandem/js/types/IOType.js';
import NumberIO from '../../../tandem/js/types/NumberIO.js';
import VoidIO from '../../../tandem/js/types/VoidIO.js';
import scenery from '../scenery.js';
import Node from './Node.js';

// In order to support unlinking from listening to the index property, keep an indexed map to callback functions
const map = {};

// The next index at which a callback will appear in the map. This always increments and we do reuse old indices
let index = 0;

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
    linkIndex: {
      returnType: NumberIO,
      parameterTypes: [ FunctionIO( VoidIO, [ NumberIO ] ) ],
      documentation: 'Following the PropertyIO.link pattern, subscribe for notifications when the index in the parent ' +
                     'changes, and receive a callback with the current value.  The return value is a numeric ID for use ' +
                     'with clearLinkIndex.',
      implementation: function( listener ) {

        // The callback which signifies the current index
        const callback = () => {
          const index = this.parents[ 0 ].indexOfChild( this );
          listener( index );
        };

        this.parents[ 0 ].childrenChangedEmitter.addListener( callback );
        callback();

        const myIndex = index;
        map[ myIndex ] = callback;
        index++;
        return myIndex;
      }
    },
    clearLinkIndex: {
      returnType: VoidIO,
      parameterTypes: [ NumberIO ],
      documentation: 'Unlink a listener that has been added using linkIndex, by its numerical ID (like setTimeout/clearTimeout)',
      implementation: function( index ) {
        const method = map[ index ];
        this.parents[ 0 ].childrenChangedEmitter.removeListener( method );
        delete map[ index ];
      }
    },
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