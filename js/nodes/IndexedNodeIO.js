// Copyright 2020-2021, University of Colorado Boulder

/**
 * IO Type for Nodes that can save their own index (if phetioState: true).  Can be used to customize z-order
 * or layout order.
 *
 * This IOType supports PhET-iO state, but only when every child within a Node's children array is an IndexedNodeIO
 * and is stateful (`phetioState: true`). This applyState algorithm uses Node "swaps" instead of index-based inserts
 * to ensure that by the end of state setting, all Nodes are in the correct order.
 * see https://github.com/phetsims/scenery/issues/1252#issuecomment-888014859 for more information.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */

import FunctionIO from '../../../tandem/js/types/FunctionIO.js';
import IOType from '../../../tandem/js/types/IOType.js';
import NullableIO from '../../../tandem/js/types/NullableIO.js';
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
      assert && assert( node.parents.length === 1, 'IndexedNodeIO only supports nodes with a single parent' );
      stateObject.index = node.parents[ 0 ].indexOfChild( node );
    }
    else {
      stateObject.index = null;
    }
    return stateObject;
  },
  applyState: ( node, stateObject ) => {
    const nodeParent = node.parents[ 0 ];

    if ( nodeParent && stateObject.index ) {
      assert && assert( node.parents.length === 1, 'IndexedNodeIO only supports nodes with a single parent' );

      // Swap the child at the destination index with current position of this Node, that way the operation is atomic.
      // This implementation assumes that all children are instrumented IndexedNodeIO instances and can have state set
      // on them to "fix them" after this operation. Without this implementation, using Node.moveChildToIndex could blow
      // away another IndexedNode state set. See https://github.com/phetsims/ph-scale/issues/227
      const children = nodeParent.children;
      const currentIndex = nodeParent.indexOfChild( node );
      children[ currentIndex ] = children[ stateObject.index ];
      children[ stateObject.index ] = node;
      nodeParent.setChildren( children );
    }
  },
  stateSchema: {
    index: NullableIO( NumberIO )
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
          assert && assert( this.parents.length === 1, 'IndexedNodeIO only supports nodes with a single parent' );
          const index = this.parents[ 0 ].indexOfChild( this );
          listener( index );
        };

        assert && assert( this.parents.length === 1, 'IndexedNodeIO only supports nodes with a single parent' );
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
        assert && assert( this.parents.length === 1, 'IndexedNodeIO only supports nodes with a single parent' );
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