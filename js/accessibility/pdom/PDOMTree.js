// Copyright 2018-2023, University of Colorado Boulder

/**
 * The main logic for maintaining the PDOM instance tree (see https://github.com/phetsims/scenery-phet/issues/365)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import arrayDifference from '../../../../phet-core/js/arrayDifference.js';
import { BrowserEvents, Node, PartialPDOMTrail, PDOMInstance, scenery, Trail } from '../../imports.js';

// Reference to the focused DOM element, so we can restore focus between big tree operations.
let activeElementId = null;

const PDOMTree = {
  /**
   * Called when a child node is added to a parent node (and the child is likely to have pdom content).
   * @public
   *
   * @param {Node} parent
   * @param {Node} child
   */
  addChild( parent, child ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `addChild parent:n#${parent._id}, child:n#${child._id}` );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && assert( parent instanceof Node );
    assert && assert( child instanceof Node );
    assert && assert( !child._rendererSummary.hasNoPDOM() );

    PDOMTree.beforeOp();

    if ( !child._pdomParent ) {
      PDOMTree.addTree( parent, child );
    }

    PDOMTree.afterOp();

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Called when a child node is removed from a parent node (and the child is likely to have pdom content).
   * @public
   *
   * @param {Node} parent
   * @param {Node} child
   */
  removeChild( parent, child ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `removeChild parent:n#${parent._id}, child:n#${child._id}` );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && assert( parent instanceof Node );
    assert && assert( child instanceof Node );
    assert && assert( !child._rendererSummary.hasNoPDOM() );

    PDOMTree.beforeOp();

    if ( !child._pdomParent ) {
      PDOMTree.removeTree( parent, child );
    }

    PDOMTree.afterOp();

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Called when a node's children are reordered (no additions/removals).
   * @public
   *
   * @param {Node} node
   */
  childrenOrderChange( node ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `childrenOrderChange node:n#${node._id}` );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && assert( node instanceof Node );
    assert && assert( !node._rendererSummary.hasNoPDOM() );

    PDOMTree.beforeOp();

    PDOMTree.reorder( node );

    PDOMTree.afterOp();

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Called when a node has an pdomOrder change.
   * @public
   *
   * @param {Node} node
   * @param {Array.<Node|null>|null} oldOrder
   * @param {Array.<Node|null>|null} newOrder
   */
  pdomOrderChange( node, oldOrder, newOrder ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `pdomOrderChange n#${node._id}: ${PDOMTree.debugOrder( oldOrder )},${PDOMTree.debugOrder( newOrder )}` );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && assert( node instanceof Node );

    PDOMTree.beforeOp();

    const removedItems = []; // {Array.<Node|null>} - May contain the placeholder null
    const addedItems = []; // {Array.<Node|null>} - May contain the placeholder null

    arrayDifference( oldOrder || [], newOrder || [], removedItems, addedItems );

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `removed: ${PDOMTree.debugOrder( removedItems )}` );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `added: ${PDOMTree.debugOrder( addedItems )}` );

    let i;
    let j;

    // Check some initial conditions
    if ( assert ) {
      for ( i = 0; i < removedItems; i++ ) {
        assert( removedItems[ i ] === null || removedItems[ i ]._pdomParent === node,
          'Node should have had an pdomOrder' );
      }
      for ( i = 0; i < addedItems; i++ ) {
        assert( addedItems[ i ] === null || addedItems[ i ]._pdomParent === null,
          'Node is already specified in an pdomOrder' );
      }
    }

    // NOTE: Performance could be improved in some cases if we can avoid rebuilding an pdom tree for DIRECT children
    // when changing whether they are present in the pdomOrder. Basically, if something is a child and NOT
    // in an pdomOrder, changing its parent's order to include it (or vice versa) triggers a rebuild when it
    // would not strictly be necessary.

    const pdomTrails = PDOMTree.findPDOMTrails( node );

    // Remove subtrees from us (that were removed)
    for ( i = 0; i < removedItems.length; i++ ) {
      const removedItemToRemove = removedItems[ i ];
      if ( removedItemToRemove ) {
        PDOMTree.removeTree( node, removedItemToRemove, pdomTrails );
        removedItemToRemove._pdomParent = null;
      }
    }

    // Remove subtrees from their parents (that will be added here instead)
    for ( i = 0; i < addedItems.length; i++ ) {
      const addedItemToRemove = addedItems[ i ];
      if ( addedItemToRemove ) {
        const removedParents = addedItemToRemove._parents;
        for ( j = 0; j < removedParents.length; j++ ) {
          PDOMTree.removeTree( removedParents[ j ], addedItemToRemove );
        }
        addedItemToRemove._pdomParent = node;
      }
    }

    // Add subtrees to their parents (that were removed from our order)
    for ( i = 0; i < removedItems.length; i++ ) {
      const removedItemToAdd = removedItems[ i ];
      if ( removedItemToAdd ) {
        const addedParents = removedItemToAdd._parents;
        for ( j = 0; j < addedParents.length; j++ ) {
          PDOMTree.addTree( addedParents[ j ], removedItemToAdd );
        }
      }
    }

    // Add subtrees to us (that were added in this order change)
    for ( i = 0; i < addedItems.length; i++ ) {
      const addedItemToAdd = addedItems[ i ];
      addedItemToAdd && PDOMTree.addTree( node, addedItemToAdd, pdomTrails );
    }

    PDOMTree.reorder( node, pdomTrails );

    PDOMTree.afterOp();

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Called when a node has an pdomContent change.
   * @public
   *
   * @param {Node} node
   */
  pdomContentChange( node ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `pdomContentChange n#${node._id}` );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && assert( node instanceof Node );

    PDOMTree.beforeOp();

    let i;
    const parents = node._pdomParent ? [ node._pdomParent ] : node._parents;
    const pdomTrailsList = []; // pdomTrailsList[ i ] := PDOMTree.findPDOMTrails( parents[ i ] )

    // For now, just regenerate the full tree. Could optimize in the future, if we can swap the content for an
    // PDOMInstance.
    for ( i = 0; i < parents.length; i++ ) {
      const parent = parents[ i ];

      const pdomTrails = PDOMTree.findPDOMTrails( parent );
      pdomTrailsList.push( pdomTrails );

      PDOMTree.removeTree( parent, node, pdomTrails );
    }

    // Do all removals before adding anything back in.
    for ( i = 0; i < parents.length; i++ ) {
      PDOMTree.addTree( parents[ i ], node, pdomTrailsList[ i ] );
    }

    // An edge case is where we change the rootNode of the display (and don't have an effective parent)
    for ( i = 0; i < node._rootedDisplays.length; i++ ) {
      const display = node._rootedDisplays[ i ];
      if ( display._accessible ) {
        PDOMTree.rebuildInstanceTree( display._rootPDOMInstance );
      }
    }

    PDOMTree.afterOp();

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Sets up a root instance with a given root node.
   * @public
   *
   * @param {PDOMInstance} rootInstance
   */
  rebuildInstanceTree( rootInstance ) {
    const rootNode = rootInstance.display.rootNode;
    assert && assert( rootNode );

    rootInstance.removeAllChildren();

    rootInstance.addConsecutiveInstances( PDOMTree.createTree( new Trail( rootNode ), rootInstance.display, rootInstance ) );
  },

  /**
   * Handles the conceptual addition of an pdom subtree.
   * @private
   *
   * @param {Node} parent
   * @param {Node} child
   * @param {Array.<PartialPDOMTrail>} [pdomTrails] - Will be computed if needed
   */
  addTree( parent, child, pdomTrails ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `addTree parent:n#${parent._id}, child:n#${child._id}` );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && PDOMTree.auditNodeForPDOMCycles( parent );

    pdomTrails = pdomTrails || PDOMTree.findPDOMTrails( parent );

    for ( let i = 0; i < pdomTrails.length; i++ ) {
      sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `trail: ${pdomTrails[ i ].trail.toString()} full:${pdomTrails[ i ].fullTrail.toString()} for ${pdomTrails[ i ].pdomInstance.toString()} root:${pdomTrails[ i ].isRoot}` );
      sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

      const partialTrail = pdomTrails[ i ];
      const parentInstance = partialTrail.pdomInstance;

      // The full trail doesn't have the child in it, so we temporarily add that for tree creation
      partialTrail.fullTrail.addDescendant( child );
      const childInstances = PDOMTree.createTree( partialTrail.fullTrail, parentInstance.display, parentInstance );
      partialTrail.fullTrail.removeDescendant( child );

      parentInstance.addConsecutiveInstances( childInstances );

      sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
    }

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Handles the conceptual removal of an pdom subtree.
   * @private
   *
   * @param {Node} parent
   * @param {Node} child
   * @param {Array.<PartialPDOMTrail>} [pdomTrails] - Will be computed if needed
   */
  removeTree( parent, child, pdomTrails ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `removeTree parent:n#${parent._id}, child:n#${child._id}` );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    pdomTrails = pdomTrails || PDOMTree.findPDOMTrails( parent );

    for ( let i = 0; i < pdomTrails.length; i++ ) {
      const partialTrail = pdomTrails[ i ];

      // The full trail doesn't have the child in it, so we temporarily add that for tree removal
      partialTrail.fullTrail.addDescendant( child );
      partialTrail.pdomInstance.removeInstancesForTrail( partialTrail.fullTrail );
      partialTrail.fullTrail.removeDescendant( child );
    }

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Handles the conceptual sorting of an pdom subtree.
   * @private
   *
   * @param {Node} node
   * @param {Array.<PartialPDOMTrail>} [pdomTrails] - Will be computed if needed
   */
  reorder( node, pdomTrails ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `reorder n#${node._id}` );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    pdomTrails = pdomTrails || PDOMTree.findPDOMTrails( node );

    for ( let i = 0; i < pdomTrails.length; i++ ) {
      const partialTrail = pdomTrails[ i ];

      // TODO: does it optimize things to pass the partial trail in (so we scan less)?
      partialTrail.pdomInstance.sortChildren();
    }

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Creates PDOM instances, returning an array of instances that should be added to the next level.
   * @private
   *
   * NOTE: Trails for which an already-existing instance exists will NOT create a new instance here. We only want to
   * fill in the "missing" structure. There are cases (a.children=[b,c], b.children=[c]) where removing an
   * pdomOrder can trigger addTree(a,b) AND addTree(b,c), and we can't create duplicate content.
   *
   * @param {Trail} trail
   * @param {Display} display
   * @param {PDOMInstance} parentInstance - Since we don't create the root here, can't be null
   * @returns {Array.<PDOMInstance>}
   */
  createTree( trail, display, parentInstance ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `createTree ${trail.toString()} parent:${parentInstance ? parentInstance.toString() : 'null'}` );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    const node = trail.lastNode();
    const effectiveChildren = node.getEffectiveChildren();

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( `effectiveChildren: ${PDOMTree.debugOrder( effectiveChildren )}` );

    // If we have pdom content ourself, we need to create the instance (so we can provide it to child instances).
    let instance;
    let existed = false;
    if ( node.hasPDOMContent ) {
      instance = parentInstance.findChildWithTrail( trail );
      if ( instance ) {
        existed = true;
      }
      else {
        instance = PDOMInstance.pool.create( parentInstance, display, trail.copy() );
      }

      // If there was an instance, then it should be the parent to effective children, otherwise, it isn't part of the
      // trail.
      parentInstance = instance;
    }

    // Create all of the direct-child instances.
    const childInstances = [];
    for ( let i = 0; i < effectiveChildren.length; i++ ) {
      trail.addDescendant( effectiveChildren[ i ], i );
      Array.prototype.push.apply( childInstances, PDOMTree.createTree( trail, display, parentInstance ) );
      trail.removeDescendant();
    }

    // If we have an instance, hook things up, and return just it.
    if ( instance ) {
      instance.addConsecutiveInstances( childInstances );

      sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
      return existed ? [] : [ instance ];
    }
    // Otherwise pass things forward so they can be added as children by the parentInstance
    else {
      sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
      return childInstances;
    }
  },

  /**
   * Prepares for a pdom-tree-changing operation (saving some state). During DOM operations we don't want Display
   * input to dispatch events as focus changes.
   * @private
   */
  beforeOp() {
    activeElementId = document.activeElement.id;
    BrowserEvents.blockFocusCallbacks = true;
  },

  /**
   * Finalizes a pdom-tree-changing operation (restoring some state).
   * @private
   */
  afterOp() {
    const activeElement = document.getElementById( activeElementId );
    activeElement && activeElement.focus();
    BrowserEvents.blockFocusCallbacks = false;
  },

  /**
   * Returns all "pdom" trails from this node ancestor-wise to nodes that have display roots.
   * @private
   *
   * NOTE: "pdom" trails may not have strict parent-child relationships between adjacent nodes, as remapping of
   * the tree can have a "PDOM parent" and "pdom child" case (the child is in the parent's pdomOrder).
   *
   * @param {Node} node
   * @returns {Array.<PartialPDOMTrail>}
   */
  findPDOMTrails( node ) {
    const trails = [];
    PDOMTree.recursivePDOMTrailSearch( trails, new Trail( node ) );
    return trails;
  },

  /**
   * Finds all partial "pdom" trails
   * @private
   *
   * @param {Array.<PartialPDOMTrail>} trailResults - Mutated, this is how we "return" our value.
   * @param {Trail} trail - Where to start from
   */
  recursivePDOMTrailSearch( trailResults, trail ) {
    const root = trail.rootNode();
    let i;

    // If we find pdom content, our search ends here. IF it is connected to any accessible pdom displays somehow, it
    // will have pdom instances. We only care about these pdom instances, as they already have any DAG
    // deduplication applied.
    if ( root.hasPDOMContent ) {
      const instances = root.pdomInstances;

      for ( i = 0; i < instances.length; i++ ) {
        trailResults.push( new PartialPDOMTrail( instances[ i ], trail.copy(), false ) );
      }
      return;
    }
    // Otherwise check for accessible pdom displays for which our node is the rootNode.
    else {
      const rootedDisplays = root.rootedDisplays;
      for ( i = 0; i < rootedDisplays.length; i++ ) {
        const display = rootedDisplays[ i ];

        if ( display._accessible ) {
          trailResults.push( new PartialPDOMTrail( display._rootPDOMInstance, trail.copy(), true ) );
        }
      }
    }

    const parents = root._pdomParent ? [ root._pdomParent ] : root._parents;
    const parentCount = parents.length;
    for ( i = 0; i < parentCount; i++ ) {
      const parent = parents[ i ];

      trail.addAncestor( parent );
      PDOMTree.recursivePDOMTrailSearch( trailResults, trail );
      trail.removeAncestor();
    }
  },

  /**
   * Ensures that the pdomDisplays on the node (and its subtree) are accurate.
   * @public
   */
  auditPDOMDisplays( node ) {
    if ( assertSlow ) {
      if ( node._pdomDisplaysInfo.canHavePDOMDisplays() ) {

        let i;
        const displays = [];

        // Concatenation of our parents' pdomDisplays
        for ( i = 0; i < node._parents.length; i++ ) {
          Array.prototype.push.apply( displays, node._parents[ i ]._pdomDisplaysInfo.pdomDisplays );
        }

        // And concatenation of any rooted displays (that support pdom)
        for ( i = 0; i < node._rootedDisplays.length; i++ ) {
          const display = node._rootedDisplays[ i ];
          if ( display._accessible ) {
            displays.push( display );
          }
        }

        const actualArray = node._pdomDisplaysInfo.pdomDisplays.slice();
        const expectedArray = displays.slice(); // slice helps in debugging
        assertSlow( actualArray.length === expectedArray.length );

        for ( i = 0; i < expectedArray.length; i++ ) {
          for ( let j = 0; j < actualArray.length; j++ ) {
            if ( expectedArray[ i ] === actualArray[ j ] ) {
              expectedArray.splice( i, 1 );
              actualArray.splice( j, 1 );
              i--;
              break;
            }
          }
        }

        assertSlow( actualArray.length === 0 && expectedArray.length === 0, 'Mismatch with accessible pdom displays' );
      }
      else {
        assertSlow( node._pdomDisplaysInfo.pdomDisplays.length === 0, 'Invisible/nonaccessible things should have no displays' );
      }
    }
  },

  /**
   * Checks a given Node (with assertions) to ensure it is not part of a cycle in the combined graph with edges
   * defined by "there is a parent-child or pdomParent-pdomOrder" relationship between the two nodes.
   * @public (scenery-internal)
   *
   * See https://github.com/phetsims/scenery/issues/787 for more information (and for some detail on the cases
   * that we want to catch).
   *
   * @param {Node} node
   */
  auditNodeForPDOMCycles( node ) {
    if ( assert ) {
      const trail = new Trail( node );

      ( function recursiveSearch() {
        const root = trail.rootNode();

        assert( trail.length <= 1 || root !== node,
          `${'Accessible PDOM graph cycle detected. The combined scene-graph DAG with pdomOrder defining additional ' +
             'parent-child relationships should still be a DAG. Cycle detected with the trail: '}${trail.toString()
          } path: ${trail.toPathString()}` );

        const parentCount = root._parents.length;
        for ( let i = 0; i < parentCount; i++ ) {
          const parent = root._parents[ i ];

          trail.addAncestor( parent );
          recursiveSearch();
          trail.removeAncestor();
        }
        // Only visit the pdomParent if we didn't already visit it as a parent.
        if ( root._pdomParent && !root._pdomParent.hasChild( root ) ) {
          trail.addAncestor( root._pdomParent );
          recursiveSearch();
          trail.removeAncestor();
        }
      } )();
    }
  },

  /**
   * Returns a string representation of an order (using Node ids) for debugging.
   * @private
   *
   * @param {Array.<Node|null>|null} pdomOrder
   * @returns {string}
   */
  debugOrder( pdomOrder ) {
    if ( pdomOrder === null ) { return 'null'; }

    return `[${pdomOrder.map( nodeOrNull => nodeOrNull === null ? 'null' : nodeOrNull._id ).join( ',' )}]`;
  }
};

scenery.register( 'PDOMTree', PDOMTree );

export default PDOMTree;