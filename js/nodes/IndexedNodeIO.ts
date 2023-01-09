// Copyright 2020-2023, University of Colorado Boulder

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
import { Node, scenery } from '../imports.js';

// In order to support unlinking from listening to the index property, keep an indexed map to callback functions
const map: Record<number, () => void> = {};

// The next index at which a callback will appear in the map. This always increments and we do reuse old indices
let index = 0;

function getAncestor( node: Node, containerDepth: number ): Node {
  for ( let i = 0; i < containerDepth; i++ ) {
    node = node.parents[ 0 ];
  }
  return node;
}

function getContainer( node: Node, containerDepth: number ): Node {
  return getAncestor( node, containerDepth );
}

function getContainerChild( node: Node, containerDepth: number ): Node {
  return getAncestor( node, containerDepth - 1 );
}

function assertChildHasOneParent( node: Node, containerDepth: number ): void {
  assert && assert( getContainerChild( node, containerDepth ).parents.length === 1, 'IndexedNodeIO only supports nodes with a single parent' );
}

// TODO: These must be cached, see https://github.com/phetsims/sun/issues/814

/**
 * @param containerDepth - How far up the hierarchy to go from the displayed node to the parent that manages the index ordering
 *                       - For example, in a ComboBoxList, there is no additional nesting, so the children are direct descendants
 *                       - of the container and hence the containerDepth is 1
 *                       - However, in Carousel, a nested AlignBox layer is used, so to get from a child node to the ordering
 *                       - container, you must go up 2 levels, so the containerDepth is 2.
 *                       NOTE: This value is an implementation detail which is not tracked in the PhET-iO API in any way
 *                       NOTE: (via type name or via state values). So it is important that it work correctly across different versions.
 *                       NOTE: Please ensure item ordering is preserved when migrating from an old to a new version.
 * @constructor
 */
function IndexedNodeIO( containerDepth = 1 ): IOType {

  assert && assert( containerDepth >= 1, 'parents must be at least one level up' );

  // Note: the containerDepth is an implementation detail that should not appear in the type names or API.
  // When changing the scene graph structure, the containerDepth should be changed accordingly, so that everything balances.
  return new IOType( 'IndexedNodeIO', {
    valueType: Node,
    documentation: 'Node that can be moved forward/back by index, which specifies z-order and/or layout order',
    supertype: Node.NodeIO,
    toStateObject: node => {
      const stateObject: { index: number | null } = { index: null };

      // Only work if the node is attached to the scene graph
      if ( node.parents[ 0 ] ) {
        assertChildHasOneParent( node, containerDepth );
        stateObject.index = getContainer( node, containerDepth ).indexOfChild( getContainerChild( node, containerDepth ) );
      }
      return stateObject;
    },
    applyState: ( node, stateObject ) => {

      // Only work if the node is attached to the scene graph
      if ( node.parents[ 0 ] && stateObject.index ) {
        assertChildHasOneParent( node, containerDepth );

        // Swap the child at the destination index with current position of this Node, that way the operation is atomic.
        // This implementation assumes that all children are instrumented IndexedNodeIO instances and can have state set
        // on them to "fix them" after this operation. Without this implementation, using Node.moveChildToIndex could blow
        // away another IndexedNode state set. See https://github.com/phetsims/ph-scale/issues/227
        const container = getContainer( node, containerDepth );

        const children = container.children;
        const currentIndex = container.indexOfChild( getContainerChild( node, containerDepth ) );
        children[ currentIndex ] = children[ stateObject.index ];
        children[ stateObject.index ] = getContainerChild( node, containerDepth );
        container.setChildren( children );
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
        implementation: function( this: Node, listener ) {

          // The callback which signifies the current index
          const callback = () => {
            assertChildHasOneParent( this, containerDepth );
            const index = getContainer( this, containerDepth ).indexOfChild( getContainerChild( this, containerDepth ) );
            listener( index );
          };

          assertChildHasOneParent( this, containerDepth );
          getContainer( this, containerDepth ).childrenChangedEmitter.addListener( callback );
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
        implementation: function( this: Node, index ) {
          const method = map[ index ];
          assertChildHasOneParent( this, containerDepth );
          getContainer( this, containerDepth ).childrenChangedEmitter.removeListener( method );
          delete map[ index ];
        }
      },
      moveForward: {
        returnType: VoidIO,
        parameterTypes: [],
        implementation: function( this: Node ) {
          return getContainerChild( this, containerDepth ).moveForward();
        },
        documentation: 'Move this node one index forward in each of its parents.  If the node is already at the front, this is a no-op.'
      },

      moveBackward: {
        returnType: VoidIO,
        parameterTypes: [],
        implementation: function( this: Node ) {
          return getContainerChild( this, containerDepth ).moveBackward();
        },
        documentation: 'Move this node one index backward in each of its parents.  If the node is already at the back, this is a no-op.'
      }
    }
  } );
}

scenery.register( 'IndexedNodeIO', IndexedNodeIO );
export default IndexedNodeIO;