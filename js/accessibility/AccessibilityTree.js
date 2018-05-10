// Copyright 2018, University of Colorado Boulder

/**
 * The main logic for maintaining the accessible instance tree (see https://github.com/phetsims/scenery-phet/issues/365)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  // modules
  var AccessibleInstance = require( 'SCENERY/accessibility/AccessibleInstance' );
  var PartialAccessibleTrail = require( 'SCENERY/accessibility/PartialAccessibleTrail' );
  var scenery = require( 'SCENERY/scenery' );
  // commented out so Require.js doesn't balk at the circular dependency
  // var Trail = require( 'SCENERY/util/Trail' );

  // constants
  var DEBUG_ORDER = function( accessibleOrder ) {
    return '[' + accessibleOrder.map( function( nodeOrNull ) {
      return nodeOrNull === null ? 'null' : nodeOrNull._id;
    } ).join( ',' ) + ']';
  };

  // globals (for restoring focus)
  var focusedNode = null;

  var AccessibilityTree = {
    /**
     * Called when a child node is added to a parent node (and the child is likely to have accessible content).
     * @public
     *
     * @param {Node} parent
     * @param {Node} child
     */
    addChild: function( parent, child ) {
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'addChild parent:n#' + parent._id + ', child:n#' + child._id );
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.push();

      assert && assert( parent instanceof scenery.Node );
      assert && assert( child instanceof scenery.Node );
      assert && assert( !child._rendererSummary.isNotAccessible() );

      AccessibilityTree.beforeOp();

      if ( !child._accessibleParent ) {
        AccessibilityTree.addTree( parent, child );
      }

      AccessibilityTree.afterOp();

      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.pop();
    },

    /**
     * Called when a child node is removed from a parent node (and the child is likely to have accessible content).
     * @public
     *
     * @param {Node} parent
     * @param {Node} child
     */
    removeChild: function( parent, child ) {
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'removeChild parent:n#' + parent._id + ', child:n#' + child._id );
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.push();

      assert && assert( parent instanceof scenery.Node );
      assert && assert( child instanceof scenery.Node );
      assert && assert( !child._rendererSummary.isNotAccessible() );

      AccessibilityTree.beforeOp();

      if ( !child._accessibleParent ) {
        AccessibilityTree.removeTree( parent, child );
      }

      AccessibilityTree.afterOp();

      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.pop();
    },

    /**
     * Called when a node has an accessibleOrder change.
     * @public
     *
     * @param {Node} node
     * @param {Array.<Node|null>} oldOrder
     * @param {Array.<Node|null>} newOrder
     */
    accessibleOrderChange: function( node, oldOrder, newOrder ) {
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'accessibleOrderChange n#' + node._id + ': ' + DEBUG_ORDER( oldOrder ) + ',' + DEBUG_ORDER( newOrder ) );
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.push();

      assert && assert( node instanceof scenery.Node );

      AccessibilityTree.beforeOp();

      var removedItems = []; // {Array.<Node|null>} - May contain the placeholder null
      var addedItems = []; // {Array.<Node|null>} - May contain the placeholder null

      AccessibilityTree.arrayDifference( oldOrder, newOrder, removedItems, addedItems );

      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'removed: ' + DEBUG_ORDER( removedItems ) );
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'added: ' + DEBUG_ORDER( addedItems ) );

      var i;
      var j;

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

      var accessibleTrails = AccessibilityTree.findAccessibleTrails( node );

      // Remove subtrees from us (that were removed)
      for ( i = 0; i < removedItems.length; i++ ) {
        var removedItemToRemove = removedItems[ i ];
        if ( removedItemToRemove ) {
          AccessibilityTree.removeTree( node, removedItemToRemove, accessibleTrails );
          removedItemToRemove._accessibleParent = null;
        }
      }

      // Remove subtrees from their parents (that will be added here instead)
      for ( i = 0; i < addedItems.length; i++ ) {
        var addedItemToRemove = addedItems[ i ];
        if ( addedItemToRemove ) {
          var removedParents = addedItemToRemove._parents;
          for ( j = 0; j < removedParents.length; j++ ) {
            AccessibilityTree.removeTree( removedParents[ j ], addedItemToRemove );
          }
          addedItemToRemove._accessibleParent = node;
        }
      }

      // Add subtrees to their parents (that were removed from our order)
      for ( i = 0; i < removedItems.length; i++ ) {
        var removedItemToAdd = removedItems[ i ];
        if ( removedItemToAdd ) {
          var addedParents = removedItemToAdd._parents;
          for ( j = 0; j < addedParents.length; j++ ) {
            AccessibilityTree.addTree( addedParents[ j ], removedItemToAdd );
          }
        }
      }

      // Add subtrees to us (that were added in this order change)
      for ( i = 0; i < addedItems.length; i++ ) {
        var addedItemToAdd = addedItems[ i ];
        addedItemToAdd && AccessibilityTree.addTree( node, addedItemToAdd, accessibleTrails );
      }

      AccessibilityTree.reorder( node, accessibleTrails );

      AccessibilityTree.afterOp();

      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.pop();
    },

    /**
     * Called when a node has an accessibleContent change.
     * @public
     *
     * @param {Node} node
     * @param {Object|null} oldContent
     * @param {Object|null} newContent
     */
    accessibleContentChange: function( node, oldContent, newContent ) {
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'accessibleContentChange n#' + node._id + ': had:' + ( oldContent !== null ) + ', has:' + ( newContent !== null ) );
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.push();

      assert && assert( node instanceof scenery.Node );

      AccessibilityTree.beforeOp();

      var parents = node._accessibleParent ? [ node._accessibleParent ] : node._parents;

      for ( var i = 0; i < parents.length; i++ ) {
        var parent = parents[ i ];

        var accessibleTrails = AccessibilityTree.findAccessibleTrails( parent );

        // For now, just regenerate the full tree. Could optimize in the future, if we can swap the content for an
        // AccessibleInstance.
        AccessibilityTree.removeTree( parent, node, accessibleTrails );
        AccessibilityTree.addTree( parent, node, accessibleTrails );
      }

      AccessibilityTree.afterOp();

      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.pop();
    },

    /**
     * Sets up a root instance with a given root node.
     * @public
     *
     * @param {AccessibleInstance} rootInstance
     * @param {Node} rootNode
     */
    initializeRoot: function( rootInstance, rootNode ) {
      rootInstance.addConsecutiveInstances( AccessibilityTree.createTree( new scenery.Trail( rootNode ), rootInstance.display, rootInstance ) );
    },

    /**
     * Handles the conceptual addition of an accessible subtree.
     * @private
     *
     * @param {Node} parent
     * @param {Node} child
     * @param {Array.<PartialAccessibleTrail>} [accessibleTrails] - Will be computed if needed
     */
    addTree: function( parent, child, accessibleTrails ) {
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'addTree parent:n#' + parent._id + ', child:n#' + child._id );
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.push();

      accessibleTrails = accessibleTrails || AccessibilityTree.findAccessibleTrails( parent );

      for ( var i = 0; i < accessibleTrails.length; i++ ) {
        sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'trail: ' + accessibleTrails[ i ].trail.toString() + ' full:' + accessibleTrails[ i ].fullTrail.toString() + ' for ' + accessibleTrails[ i ].accessibleInstance.toString() + ' root:' + accessibleTrails[ i ].isRoot );
        sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.push();

        var partialTrail = accessibleTrails[ i ];
        var parentInstance = partialTrail.accessibleInstance;

        // The full trail doesn't have the child in it, so we temporarily add that for tree creation
        partialTrail.fullTrail.addDescendant( child );
        var childInstances = AccessibilityTree.createTree( partialTrail.fullTrail, parentInstance.display, parentInstance );
        partialTrail.fullTrail.removeDescendant( child );

        parentInstance.addConsecutiveInstances( childInstances );

        sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.pop();
      }

      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.pop();
    },

    /**
     * Handles the conceptual removal of an accessible subtree.
     * @private
     *
     * @param {Node} parent
     * @param {Node} child
     * @param {Array.<PartialAccessibleTrail>} [accessibleTrails] - Will be computed if needed
     */
    removeTree: function( parent, child, accessibleTrails ) {
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'removeTree parent:n#' + parent._id + ', child:n#' + child._id );
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.push();

      accessibleTrails = accessibleTrails || AccessibilityTree.findAccessibleTrails( parent );

      for ( var i = 0; i < accessibleTrails.length; i++ ) {
        var partialTrail = accessibleTrails[ i ];

        partialTrail.accessibleInstance.removeInstancesForTrail( partialTrail.fullTrail );
      }

      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.pop();
    },

    /**
     * Handles the conceptual sorting of an accessible subtree.
     * @private
     *
     * @param {Node} node
     * @param {Array.<PartialAccessibleTrail>} [accessibleTrails] - Will be computed if needed
     */
    reorder: function( node, accessibleTrails ) {
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'reorder n#' + node._id );
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.push();

      accessibleTrails = accessibleTrails || AccessibilityTree.findAccessibleTrails( node );

      for ( var i = 0; i < accessibleTrails.length; i++ ) {
        var partialTrail = accessibleTrails[ i ];

        // TODO: does it optimize things to pass the partial trail in (so we scan less)?
        partialTrail.accessibleInstance.sortChildren();
      }

      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.pop();
    },

    /**
     * Creates accessible instances, returning an array of instances that should be added to the next level.
     * @private
     *
     * @param {Trail} trail
     * @param {Display} display
     * @param {AccessibleInstance} parentInstance - Since we don't create the root here, can't be null
     * @returns {Array.<AccessibleInstance>}
     */
    createTree: function( trail, display, parentInstance ) {
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'createTree ' + trail.toString() + ' parent:' + ( parentInstance ? parentInstance.toString() : 'null' ) );
      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.push();

      var node = trail.lastNode();
      var effectiveChildren = node.getEffectiveChildren();

      sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.AccessibilityTree( 'effectiveChildren: ' + DEBUG_ORDER( effectiveChildren ) );

      // If we are accessible ourself, we need to create the instance (so we can provide it to child instances).
      var instance = null;
      if ( node.accessibleContent ) {
        instance = AccessibleInstance.createFromPool( parentInstance, display, trail.copy() );
        parentInstance = instance;
      }

      // Create all of the direct-child instances.
      var childInstances = [];
      for ( var i = 0; i < effectiveChildren.length; i++ ) {
        trail.addDescendant( effectiveChildren[ i ], i );
        Array.prototype.push.apply( childInstances, AccessibilityTree.createTree( trail, display, parentInstance ) );
        trail.removeDescendant();
      }

      // If we have an instance, hook things up, and return just it.
      if ( instance ) {
        instance.addConsecutiveInstances( childInstances );

        sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.pop();
        return [ instance ];
      }
      // Otherwise pass things forward so they can be added as children by the parentInstance
      else {
        sceneryLog && sceneryLog.AccessibilityTree && sceneryLog.pop();
        return childInstances;
      }
    },

    /**
     * Prepares for an a11y-tree-changing operation (saving some state).
     * @private
     */
    beforeOp: function() {
      // paranoia about initialization order (should be safe)
      focusedNode = scenery.Display && scenery.Display.focusedNode;
    },

    /**
     * Finalizes an a11y-tree-changing operation (restoring some state)
     * @private
     */
    afterOp: function() {
      focusedNode && focusedNode.focus();
    },

    /**
     * Given two arrays, find the items that are only in one of them (mutates the aOnly and bOnly parameters)
     * @public
     *
     * NOTE: Assumes there are no duplicate values in the arrays.
     *
     * TODO: Consider using this type of thing for Node children for setChildren, etc.
     *
     * @param {Array.<*>} a
     * @param {Array.<*>} b
     * @param {Array.<*>} aOnly
     * @param {Array.<*>} bOnly
     */
    arrayDifference: function( a, b, aOnly, bOnly ) {
      assert && assert( Array.isArray( a ) );
      assert && assert( Array.isArray( b ) );
      assert && assert( Array.isArray( aOnly ) && aOnly.length === 0 );
      assert && assert( Array.isArray( bOnly ) && bOnly.length === 0 );

      Array.prototype.push.apply( aOnly, a );
      Array.prototype.push.apply( bOnly, b );

      outerLoop:
      for ( var i = 0; i < aOnly.length; i++ ) {
        var aItem = aOnly[ i ];

        for ( var j = 0; j < bOnly.length; j++ ) {
          var bItem = bOnly[ j ];

          if ( aItem === bItem ) {
            aOnly.splice( i, 1 );
            bOnly.splice( j, 1 );
            j = 0;
            if ( i === aOnly.length ) {
              break outerLoop;
            }
            i -= 1;
          }
        }
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
     * @return {Array.<PartialAccessibleTrail>}
     */
    findAccessibleTrails: function( node ) {
      var trails = [];
      AccessibilityTree.recursiveAccessibleTrailSearch( trails, new scenery.Trail( node ) );
      return trails;
    },

    /**
     * Finds all partial "accessible" trails
     * @private
     *
     * @param {Array.<PartialAccessibleTrail>} trailResults - Mutated, this is how we "return" our value.
     * @param {Trail} trail - Where to start from
     */
    recursiveAccessibleTrailSearch: function( trailResults, trail ) {
      var root = trail.rootNode();
      var i;

      // If we find accessible content, our search ends here. IF it is connected to any accessible displays somehow, it
      // will have accessible instances. We only care about these accessible instances, as they already have any DAG
      // deduplication applied.
      if ( root.accessibleContent ) {
        var instances = root.accessibleInstances;

        for ( i = 0; i < instances.length; i++ ) {
          trailResults.push( new PartialAccessibleTrail( instances[ i ], trail.copy(), false ) );
        }
        return;
      }
      // Otherwise check for accessible displays for which our node is the rootNode.
      else {
        var rootedDisplays = root.rootedDisplays;
        for ( i = 0; i < rootedDisplays.length; i++ ) {
          var display = rootedDisplays[ i ];

          if ( display._accessible ) {
            trailResults.push( new PartialAccessibleTrail( display._rootAccessibleInstance, trail.copy(), true ) );
          }
        }
      }

      var parents = root._accessibleParent ? [ root._accessibleParent ] : root._parents;
      var parentCount = parents.length;
      for ( i = 0; i < parentCount; i++ ) {
        var parent = parents[ i ];

        trail.addAncestor( parent );
        AccessibilityTree.recursiveAccessibleTrailSearch( trailResults, trail );
        trail.removeAncestor();
      }
    }
  };

  scenery.register( 'AccessibilityTree', AccessibilityTree );

  return AccessibilityTree;
} );
