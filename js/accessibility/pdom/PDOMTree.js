// Copyright 2018-2020, University of Colorado Boulder

/**
 * The main logic for maintaining the accessible instance tree (see https://github.com/phetsims/scenery-phet/issues/365)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import arrayDifference from '../../../../phet-core/js/arrayDifference.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import Trail from '../../util/Trail.js';
import PDOMInstance from './PDOMInstance.js';
import PartialPDOMTrail from './PartialPDOMTrail.js';

// globals (for restoring focus)
let focusedNode = null;

var PDOMTree = {
  /**
   * Called when a child node is added to a parent node (and the child is likely to have accessible content).
   * @public
   *
   * @param {Node} parent
   * @param {Node} child
   */
  addChild( parent, child ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'addChild parent:n#' + parent._id + ', child:n#' + child._id );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && assert( parent instanceof Node );
    assert && assert( child instanceof Node );
    assert && assert( !child._rendererSummary.isNotAccessible() );

    const blockedDisplays = PDOMTree.beforeOp( child );

    if ( !child._accessibleParent ) {
      PDOMTree.addTree( parent, child );
    }

    PDOMTree.afterOp( blockedDisplays );

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Called when a child node is removed from a parent node (and the child is likely to have accessible content).
   * @public
   *
   * @param {Node} parent
   * @param {Node} child
   */
  removeChild( parent, child ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'removeChild parent:n#' + parent._id + ', child:n#' + child._id );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && assert( parent instanceof Node );
    assert && assert( child instanceof Node );
    assert && assert( !child._rendererSummary.isNotAccessible() );

    const blockedDisplays = PDOMTree.beforeOp( child );

    if ( !child._accessibleParent ) {
      PDOMTree.removeTree( parent, child );
    }

    PDOMTree.afterOp( blockedDisplays );

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Called when a node's children are reordered (no additions/removals).
   * @public
   *
   * @param {Node} node
   */
  childrenOrderChange( node ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'childrenOrderChange node:n#' + node._id );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && assert( node instanceof Node );
    assert && assert( !node._rendererSummary.isNotAccessible() );

    const blockedDisplays = PDOMTree.beforeOp( node );

    PDOMTree.reorder( node );

    PDOMTree.afterOp( blockedDisplays );

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Called when a node has an accessibleOrder change.
   * @public
   *
   * @param {Node} node
   * @param {Array.<Node|null>|null} oldOrder
   * @param {Array.<Node|null>|null} newOrder
   */
  accessibleOrderChange( node, oldOrder, newOrder ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'accessibleOrderChange n#' + node._id + ': ' + PDOMTree.debugOrder( oldOrder ) + ',' + PDOMTree.debugOrder( newOrder ) );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && assert( node instanceof Node );

    const blockedDisplays = PDOMTree.beforeOp( node );

    const removedItems = []; // {Array.<Node|null>} - May contain the placeholder null
    const addedItems = []; // {Array.<Node|null>} - May contain the placeholder null

    arrayDifference( oldOrder || [], newOrder || [], removedItems, addedItems );

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'removed: ' + PDOMTree.debugOrder( removedItems ) );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'added: ' + PDOMTree.debugOrder( addedItems ) );

    let i;
    let j;

    // Check some initial conditions
    if ( assert ) {
      for ( i = 0; i < removedItems; i++ ) {
        assert( removedItems[ i ] === null || removedItems[ i ]._accessibleParent === node,
          'Node should have had an accessibleOrder' );
      }
      for ( i = 0; i < addedItems; i++ ) {
        assert( addedItems[ i ] === null || addedItems[ i ]._accessibleParent === null,
          'Node is already specified in an accessibleOrder' );
      }
    }

    // NOTE: Performance could be improved in some cases if we can avoid rebuilding an a11y tree for DIRECT children
    // when changing whether they are present in the accessibleOrder. Basically, if something is a child and NOT
    // in an accessibleOrder, changing its parent's order to include it (or vice versa) triggers a rebuild when it
    // would not strictly be necessary.

    const accessibleTrails = PDOMTree.findAccessibleTrails( node );

    // Remove subtrees from us (that were removed)
    for ( i = 0; i < removedItems.length; i++ ) {
      const removedItemToRemove = removedItems[ i ];
      if ( removedItemToRemove ) {
        PDOMTree.removeTree( node, removedItemToRemove, accessibleTrails );
        removedItemToRemove._accessibleParent = null;
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
        addedItemToRemove._accessibleParent = node;
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
      addedItemToAdd && PDOMTree.addTree( node, addedItemToAdd, accessibleTrails );
    }

    PDOMTree.reorder( node, accessibleTrails );

    PDOMTree.afterOp( blockedDisplays );

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Called when a node has an accessibleContent change.
   * @public
   *
   * @param {Node} node
   */
  accessibleContentChange( node ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'accessibleContentChange n#' + node._id );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && assert( node instanceof Node );

    const blockedDisplays = PDOMTree.beforeOp( node );

    let i;
    const parents = node._accessibleParent ? [ node._accessibleParent ] : node._parents;
    const accessibleTrailsList = []; // accessibleTrailsList[ i ] := AccessibilityTree.findAccessibleTrails( parents[ i ] )

    // For now, just regenerate the full tree. Could optimize in the future, if we can swap the content for an
    // PDOMInstance.
    for ( i = 0; i < parents.length; i++ ) {
      const parent = parents[ i ];

      const accessibleTrails = PDOMTree.findAccessibleTrails( parent );
      accessibleTrailsList.push( accessibleTrails );

      PDOMTree.removeTree( parent, node, accessibleTrails );
    }

    // Do all removals before adding anything back in.
    for ( i = 0; i < parents.length; i++ ) {
      PDOMTree.addTree( parents[ i ], node, accessibleTrailsList[ i ] );
    }

    // An edge case is where we change the rootNode of the display (and don't have an effective parent)
    for ( i = 0; i < node._rootedDisplays.length; i++ ) {
      const display = node._rootedDisplays[ i ];
      if ( display._accessible ) {
        PDOMTree.rebuildInstanceTree( display._rootAccessibleInstance );
      }
    }

    PDOMTree.afterOp( blockedDisplays );

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
   * Handles the conceptual addition of an accessible subtree.
   * @private
   *
   * @param {Node} parent
   * @param {Node} child
   * @param {Array.<PartialPDOMTrail>} [accessibleTrails] - Will be computed if needed
   */
  addTree( parent, child, accessibleTrails ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'addTree parent:n#' + parent._id + ', child:n#' + child._id );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    assert && PDOMTree.auditNodeForAccessibleCycles( parent );

    accessibleTrails = accessibleTrails || PDOMTree.findAccessibleTrails( parent );

    for ( let i = 0; i < accessibleTrails.length; i++ ) {
      sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'trail: ' + accessibleTrails[ i ].trail.toString() + ' full:' + accessibleTrails[ i ].fullTrail.toString() + ' for ' + accessibleTrails[ i ].accessibleInstance.toString() + ' root:' + accessibleTrails[ i ].isRoot );
      sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

      const partialTrail = accessibleTrails[ i ];
      const parentInstance = partialTrail.accessibleInstance;

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
   * Handles the conceptual removal of an accessible subtree.
   * @private
   *
   * @param {Node} parent
   * @param {Node} child
   * @param {Array.<PartialPDOMTrail>} [accessibleTrails] - Will be computed if needed
   */
  removeTree( parent, child, accessibleTrails ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'removeTree parent:n#' + parent._id + ', child:n#' + child._id );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    accessibleTrails = accessibleTrails || PDOMTree.findAccessibleTrails( parent );

    for ( let i = 0; i < accessibleTrails.length; i++ ) {
      const partialTrail = accessibleTrails[ i ];

      // The full trail doesn't have the child in it, so we temporarily add that for tree removal
      partialTrail.fullTrail.addDescendant( child );
      partialTrail.accessibleInstance.removeInstancesForTrail( partialTrail.fullTrail );
      partialTrail.fullTrail.removeDescendant( child );
    }

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Handles the conceptual sorting of an accessible subtree.
   * @private
   *
   * @param {Node} node
   * @param {Array.<PartialPDOMTrail>} [accessibleTrails] - Will be computed if needed
   */
  reorder( node, accessibleTrails ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'reorder n#' + node._id );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    accessibleTrails = accessibleTrails || PDOMTree.findAccessibleTrails( node );

    for ( let i = 0; i < accessibleTrails.length; i++ ) {
      const partialTrail = accessibleTrails[ i ];

      // TODO: does it optimize things to pass the partial trail in (so we scan less)?
      partialTrail.accessibleInstance.sortChildren();
    }

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.pop();
  },

  /**
   * Creates accessible instances, returning an array of instances that should be added to the next level.
   * @private
   *
   * NOTE: Trails for which an already-existing instance exists will NOT create a new instance here. We only want to
   * fill in the "missing" structure. There are cases (a.children=[b,c], b.children=[c]) where removing an
   * accessibleOrder can trigger addTree(a,b) AND addTree(b,c), and we can't create duplicate content.
   *
   * @param {Trail} trail
   * @param {Display} display
   * @param {PDOMInstance} parentInstance - Since we don't create the root here, can't be null
   * @returns {Array.<PDOMInstance>}
   */
  createTree( trail, display, parentInstance ) {
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'createTree ' + trail.toString() + ' parent:' + ( parentInstance ? parentInstance.toString() : 'null' ) );
    sceneryLog && sceneryLog.PDOMTree && sceneryLog.push();

    const node = trail.lastNode();
    const effectiveChildren = node.getEffectiveChildren();

    sceneryLog && sceneryLog.PDOMTree && sceneryLog.PDOMTree( 'effectiveChildren: ' + PDOMTree.debugOrder( effectiveChildren ) );

    // If we are accessible ourself, we need to create the instance (so we can provide it to child instances).
    let instance;
    let existed = false;
    if ( node.hasPDOMContent ) {
      instance = parentInstance.findChildWithTrail( trail );
      if ( instance ) {
        existed = true;
      }
      else {
        instance = PDOMInstance.createFromPool( parentInstance, display, trail.copy() );
      }
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
   * Prepares for an a11y-tree-changing operation (saving some state). During DOM operations we don't want Display
   * input to dispatch events as focus changes.
   * @private
   *
   * @param {Node} node - root of Node subtree whose PDOMInstance tree is being rearranged.
   */
  beforeOp( node ) {
    // paranoia about initialization order (should be safe)
    focusedNode = scenery.Display && scenery.Display.focusedNode;

    // list of displays to stop blocking focus callbacks in afterOp
    const displays = [];

    const accessibleTrails = this.findAccessibleTrails( node );
    for ( let i = 0; i < accessibleTrails.length; i++ ) {
      const display = accessibleTrails[ i ].accessibleInstance.display;
      display.blockFocusCallbacks = true;
      displays.push( display );
    }

    return displays;
  },

  /**
   * Finalizes an a11y-tree-changing operation (restoring some state).
   * @private
   *
   * @param {Array.<Display>} blockedDisplays
   */
  afterOp( blockedDisplays ) {
    focusedNode && focusedNode.focus();

    for ( let i = 0; i < blockedDisplays.length; i++ ) {
      blockedDisplays[ i ].blockFocusCallbacks = false;
    }
  },

  /**
   * Returns all "accessible" trails from this node ancestor-wise to nodes that have display roots.
   * @private
   *
   * NOTE: "accessible" trails may not have strict parent-child relationships between adjacent nodes, as remapping of
   * the tree can have an "accessible parent" and "accessible child" case (the child is in the parent's
   * accessibleOrder).
   *
   * @param {Node} node
   * @returns {Array.<PartialPDOMTrail>}
   */
  findAccessibleTrails( node ) {
    const trails = [];
    PDOMTree.recursiveAccessibleTrailSearch( trails, new Trail( node ) );
    return trails;
  },

  /**
   * Finds all partial "accessible" trails
   * @private
   *
   * @param {Array.<PartialPDOMTrail>} trailResults - Mutated, this is how we "return" our value.
   * @param {Trail} trail - Where to start from
   */
  recursiveAccessibleTrailSearch( trailResults, trail ) {
    const root = trail.rootNode();
    let i;

    // If we find accessible content, our search ends here. IF it is connected to any accessible displays somehow, it
    // will have accessible instances. We only care about these accessible instances, as they already have any DAG
    // deduplication applied.
    if ( root.hasPDOMContent ) {
      const instances = root.accessibleInstances;

      for ( i = 0; i < instances.length; i++ ) {
        trailResults.push( new PartialPDOMTrail( instances[ i ], trail.copy(), false ) );
      }
      return;
    }
    // Otherwise check for accessible displays for which our node is the rootNode.
    else {
      const rootedDisplays = root.rootedDisplays;
      for ( i = 0; i < rootedDisplays.length; i++ ) {
        const display = rootedDisplays[ i ];

        if ( display._accessible ) {
          trailResults.push( new PartialPDOMTrail( display._rootAccessibleInstance, trail.copy(), true ) );
        }
      }
    }

    const parents = root._accessibleParent ? [ root._accessibleParent ] : root._parents;
    const parentCount = parents.length;
    for ( i = 0; i < parentCount; i++ ) {
      const parent = parents[ i ];

      trail.addAncestor( parent );
      PDOMTree.recursiveAccessibleTrailSearch( trailResults, trail );
      trail.removeAncestor();
    }
  },

  /**
   * Ensures that the accessibleDisplays on the node (and its subtree) are accurate.
   * @public
   */
  auditAccessibleDisplays( node ) {
    if ( assertSlow ) {
      if ( node._accessibleDisplaysInfo.canHaveAccessibleDisplays() ) {

        let i;
        const displays = [];

        // Concatenation of our parents' accessibleDisplays
        for ( i = 0; i < node._parents.length; i++ ) {
          Array.prototype.push.apply( displays, node._parents[ i ]._accessibleDisplaysInfo.accessibleDisplays );
        }

        // And concatenation of any rooted displays (that are a11y)
        for ( i = 0; i < node._rootedDisplays.length; i++ ) {
          const display = node._rootedDisplays[ i ];
          if ( display._accessible ) {
            displays.push( display );
          }
        }

        const actualArray = node._accessibleDisplaysInfo.accessibleDisplays.slice();
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

        assertSlow( actualArray.length === 0 && expectedArray.length === 0, 'Mismatch with accessible displays' );
      }
      else {
        assertSlow( node._accessibleDisplaysInfo.accessibleDisplays.length === 0, 'Invisible/nonaccessible things should have no displays' );
      }
    }
  },

  /**
   * Checks a given Node (with assertions) to ensure it is not part of a cycle in the combined graph with edges
   * defined by "there is a parent-child or accessibleParent-accessibleOrder" relationship between the two nodes.
   * @public (scenery-internal)
   *
   * See https://github.com/phetsims/scenery/issues/787 for more information (and for some detail on the cases
   * that we want to catch).
   *
   * @param {Node} node
   */
  auditNodeForAccessibleCycles( node ) {
    if ( assert ) {
      const trail = new Trail( node );

      ( function recursiveSearch() {
        const root = trail.rootNode();

        assert( trail.length <= 1 || root !== node,
          'Accessible graph cycle detected. The combined scene-graph DAG with accessibleOrder defining additional ' +
          'parent-child relationships should still be a DAG. Cycle detected with the trail: ' + trail.toString() +
          ' path: ' + trail.toPathString() );

        const parentCount = root._parents.length;
        for ( let i = 0; i < parentCount; i++ ) {
          const parent = root._parents[ i ];

          trail.addAncestor( parent );
          recursiveSearch();
          trail.removeAncestor();
        }
        // Only visit the accessibleParent if we didn't already visit it as a parent.
        if ( root._accessibleParent && !root._accessibleParent.hasChild( root ) ) {
          trail.addAncestor( root._accessibleParent );
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
   * @param {Array.<Node|null>|null} accessibleOrder
   * @returns {string}
   */
  debugOrder( accessibleOrder ) {
    if ( accessibleOrder === null ) { return 'null'; }

    return '[' + accessibleOrder.map( nodeOrNull => nodeOrNull === null ? 'null' : nodeOrNull._id ).join( ',' ) + ']';
  }
};

scenery.register( 'PDOMTree', PDOMTree );

export default PDOMTree;