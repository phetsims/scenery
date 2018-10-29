// Copyright 2015-2016, University of Colorado Boulder

/**
 * An instance that is synchronously created, for handling accessibility needs.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  // var AccessiblePeer = require( 'SCENERY/accessibility/AccessiblePeer' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Events = require( 'AXON/Events' );
  var FullScreen = require( 'SCENERY/util/FullScreen' );
  var inherit = require( 'PHET_CORE/inherit' );
  var KeyboardUtil = require( 'SCENERY/accessibility/KeyboardUtil' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );

  var globalId = 1;

  /**
   * @constructor
   * @mixes Poolable
   *
   * @param parent
   * @param display
   * @param trail
   */
  function AccessibleInstance( parent, display, trail ) {
    this.initializeAccessibleInstance( parent, display, trail );
  }

  scenery.register( 'AccessibleInstance', AccessibleInstance );

  inherit( Events, AccessibleInstance, {
    /**
     * @param {AccessibleInstance|null} parent
     * @param {Display} display
     * @param {HTMLElement} [domElement] - If not included here, subtype is responsible for setting it in the constructor.
     * @returns {AccessibleInstance} - Returns 'this' reference, for chaining
     */
    initializeAccessibleInstance: function( parent, display, trail ) {
      Events.call( this ); // TODO: is Events worth mixing in by default? Will we need to listen to events?

      assert && assert( !this.id || this.disposed, 'If we previously existed, we need to have been disposed' );

      // unique ID
      this.id = this.id || globalId++;

      this.parent = parent;
      this.display = display;
      this.trail = trail;
      this.node = trail.lastNode();
      this.isRootInstance = this.trail.length === 0;

      // TODO: doc
      this.children = cleanArray( this.children );

      // If we are the root accessible instance, we won't actually have a reference to a node.
      if ( this.node ) {
        this.node.addAccessibleInstance( this );
      }

      this.isSorted = true;

      if ( this.isRootInstance ) {
        var accessibilityContainer = document.createElement( 'div' );
        accessibilityContainer.className = 'accessibility';
        accessibilityContainer.style.position = 'absolute';
        accessibilityContainer.style.left = '0';
        accessibilityContainer.style.top = '0';
        accessibilityContainer.style.width = '0';
        accessibilityContainer.style.height = '0';
        accessibilityContainer.style.clip = 'rect(0,0,0,0)';
        accessibilityContainer.style.pointerEvents = 'none';
        this.peer = new scenery.AccessiblePeer( this, accessibilityContainer );

        var self = this;
        document.body.addEventListener( 'keydown', function( event ) {

          scenery.Display.userGestureEmitter.emit();

          // If navigating in full screen mode, prevent a bug where focus gets lost if fullscreen mode was initiated
          // from an iframe by keeping focus in the display. getNext/getPreviousFocusable will return active element
          // if there are no more elements in that direction. See https://github.com/phetsims/scenery/issues/883
          if ( FullScreen.isFullScreen() && event.keyCode === KeyboardUtil.KEY_TAB ) {
            var rootElement = self.display.accessibleDOMElement;
            var nextElement = event.shiftKey ? AccessibilityUtil.getPreviousFocusable( rootElement ) :
                                               AccessibilityUtil.getNextFocusable( rootElement );
            if ( nextElement === event.target ) {
              event.preventDefault();
            }
          }

          // if an accessible node was being interacted with a mouse, or had focus when sim is made inactive, this node
          // should receive focus upon resuming keyboard navigation
          if ( self.display.pointerFocus || self.display.activeNode ) {
            var active = self.display.pointerFocus || self.display.activeNode;
            var focusable = active.focusable;
            var visible = active.visible;
            var inDOM = active.getAccessibleInstances().length > 0;
            if ( focusable && visible && inDOM ) {
              if ( event.keyCode === KeyboardUtil.KEY_TAB ) {
                event.preventDefault();
                active.focus();
                self.display.pointerFocus = null;
                self.display.activeNode = null;
              }
            }
          }
        } );
      }
      else {
        this.peer = this.node.accessibleContent.createPeer( this );
        var childContainerElement = this.parent.peer.getChildContainerElement();

        // insert the peer's DOM element or its parent if it is contained in a parent element for structure
        childContainerElement.insertBefore( this.peer.getParentContainerElement(), childContainerElement.childNodes[ 0 ] );

        // get the difference between the trails of the parent and this AccessibleInstance
        var parentTrail = this.parent.trail;
        var thisTrail = this.trail;
        this.trailDiff = [];
        for ( var i = parentTrail.length; i < thisTrail.length; i++ ) {
          this.trailDiff.push( thisTrail.get( i ) );
        }

        // when visibility or accessibleVisibility of a node in between the two trails changes, we must update
        // visibility of this peer's DOM content
        this.accessibleVisibilityListener = this.updateVisibility.bind( this );
        for ( var j = 0; j < this.trailDiff.length; j++ ) {
          this.trailDiff[ j ].onStatic( 'visibility', this.accessibleVisibilityListener );
          this.trailDiff[ j ].accessibleVisibilityChangedEmitter.addListener( this.accessibleVisibilityListener );
        }
        this.accessibleVisibilityListener();
      }

      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance(
        'Initialized ' + this.toString() );

      return this;
    },

    /**
     * Consider the following example:
     *
     * We have a node structure:
     * A
     *  B ( accessible )
     *    C (accessible )
     *      D
     *        E (accessible)
     *         G (accessible)
     *        F
     *          H (accessible)
     *
     *
     * Which has an equivalent accessible instance tree:
     * root
     *  AB
     *    ABC
     *      ABCDE
     *        ABCDEG
     *      ABCDFH
     *
     * Produces the call tree for adding instances to the accessible instance tree:
     * ABC.addSubtree( ABCD ) - not accessible
     *     ABC.addSubtree( ABCDE)
     *       ABCDE.addSubtree( ABCDEG )
     *     ABC.addSubtree( ABCDF )
     *       ABC.addSubtree( ABCDFH )
     */
    addSubtree: function( trail ) {
      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance(
        'addSubtree on ' + this.toString() + ' with trail ' + trail.toString() );
      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.push();

      var node = trail.lastNode();
      var nextInstance = this; // eslint-disable-line consistent-this
      if ( node.accessibleContent ) {
        var accessibleInstance = AccessibleInstance.createFromPool( this, this.display, trail.copy() ); // TODO: Pooling
        sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance(
          'Insert parent: ' + this.toString() + ', (new) child: ' + accessibleInstance.toString() );
        this.children.push( accessibleInstance ); // TODO: Mark us as dirty for performance.
        this.markAsUnsorted();

        nextInstance = accessibleInstance;
      }
      var children = node._children;
      for ( var i = 0; i < children.length; i++ ) {
        trail.addDescendant( children[ i ], i );
        nextInstance.addSubtree( trail );
        trail.removeDescendant();
      }

      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.pop();
    },

    removeSubtree: function( trail ) {
      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance(
        'removeSubtree on ' + this.toString() + ' with trail ' + trail.toString() );
      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.push();

      for ( var i = this.children.length - 1; i >= 0; i-- ) {
        var childInstance = this.children[ i ];
        if ( childInstance.trail.isExtensionOf( trail, true ) ) {
          sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance(
            'Remove parent: ' + this.toString() + ', child: ' + childInstance.toString() );
          this.children.splice( i, 1 ); // remove it from the children array

          // Dispose the entire subtree of AccessibleInstances
          childInstance.dispose();
        }
      }

      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.pop();
    },

    /**
     * TODO: doc
     */
    markAsUnsorted: function() {
      if ( this.isSorted ) {
        this.isSorted = false;
        this.display.markUnsortedAccessibleInstance( this );
      }
    },

    /**
     * Update visibility of this peer's accessible DOM content. The hidden attribute will hide all of the descendant
     * DOM content, so it is not necessary to update the subtree of AccessibleInstances since the browser
     * will do this for us.
     *
     * @private
     */
    updateVisibility: function() {

      // if all nodes in the trail diff are both visible and accessibleVisible, this AccessibleInstance will be
      // visible for screen readers
      var visibilityCount = 0;
      for ( var i = 0; i < this.trailDiff.length; i++ ) {
        if ( this.trailDiff[ i ].visible && this.trailDiff[ i ].accessibleVisible ) {
          visibilityCount++;
        }
      }
      this.peer.getParentContainerElement().hidden = !( visibilityCount === this.trailDiff.length );
    },

    /**
     * Sort our child accessible instances in the order they should appear in the parallel DOM. We do this by
     * creating a comparison function between two accessible instances, and sorting the array with that.
     */
    sortChildren: function() {
      assert && assert( !this.isSorted, 'No need to sort children if it is already marked as sorted' );
      this.isSorted = true;

      var parentInstance = this; // eslint-disable-line consistent-this

      // Reindex trails in preparation for sorting
      for ( var m = 0; m < this.children.length; m++ ) {
        this.children[ m ].trail.reindex();
      }

      this.children.sort( function( a, b ) {
        // Sort between a {AccessibleInstance} and b {AccessibleInstance}. This is a process where we start at our
        // "parent" accessible instance's location in the trail, and walk down looking for accessible orders that may
        // determine the order between these two instances.
        // This allows ancestor orders (for instance, an order on this instance) to override descendant orders.
        var aNodes = a.trail.nodes;
        var bNodes = b.trail.nodes;

        // Starting at the Node for this accessible instance, and walking down until the trails diverge. If our loop
        // here reaches a place where they diverge, we can use document order to determine which comes first (since
        // that means we didn't hit any relevant accessible orders). If our parentInstance is the root instance, its
        // trail will have length 0 (nothing shared), so our starting index should be set to 0.
        for ( var i = Math.max( 0, parentInstance.trail.length - 1 ); aNodes[ i ] === bNodes[ i ]; i++ ) {
          var currentNode = aNodes[ i ];
          var order = currentNode.accessibleOrder;

          // If there is no order specified on this node, we want to continue to the next children in the trails.
          if ( !order ) {
            continue;
          }

          // Loop through items in the order, since the first elements in the order are the most significant.
          for ( var j = 0; j < order.length; j++ ) {
            var orderedNode = order[ j ];
            // Find where (if at all) our Node in the order is present in the trails.
            var aIndex = aNodes.indexOf( orderedNode );
            var bIndex = bNodes.indexOf( orderedNode );

            // If the ordered node is present in both trails, we need to first determine whether the trail to those
            // nodes is the same. If they are the same, we will jump ahead to that ordered node (deliberately skipping
            // any orders on nodes in-between), and continuing on from there. If they are different, the document order
            // between the two trails to the ordered node will determine which of our instances comes first.
            if ( aIndex >= 0 && bIndex >= 0 ) {
              // Determine the index at where our two trails diverge.
              var branchIndex = i + 1;
              while ( aNodes[ branchIndex ] === bNodes[ branchIndex ] ) {
                branchIndex++;
              }

              // If the index of our ordered node in one of the trails is before our branch index, it means the trails
              // to the ordered node are the same. We want to jump ahead to the ordered node's location and continue
              // with our outer for loop.
              if ( aIndex < branchIndex ) {
                // We want the next iteration of the outer for loop to start at the index of the ordered node. Since i
                // will be incremented before the next loop begins, we subtract 1 here.
                i = aIndex - 1;

                // Exit the inner for loop
                break;
              }
              // The trails to the ordered node are different. We use the document order between the two trails to
              // determine which instance is first. This can be done by inspecting just the difference at the branch
              // index.
              else {
                // Since branchIndex is the index in the trail of two different children, we want to compare the indices
                // for the parent node of the branch nodes, thus we need to subtract 1 from our branch index.
                var aChildIndex = a.trail.indices[ branchIndex - 1 ];
                var bChildIndex = b.trail.indices[ branchIndex - 1 ];
                if ( aChildIndex < bChildIndex ) {
                  return -1;
                }
                else if ( aChildIndex > bChildIndex ) {
                  return 1;
                }
                else {
                  throw new Error( 'Two different children have the same child index' );
                }
              }
            }
            // If only the first trail is under an ordered node, it is first
            else if ( aIndex >= 0 ) {
              return -1;
            }
            // If only the second trail is under an ordered node, it is first
            else if ( bIndex >= 0 ) {
              return 1;
            }
          }
        }

        // If we reach here and haven't returned, it should be a document order comparison AND the index i determines
        // the current index where the nodes are different (branch index).
        // Since i is the index in the trail of two different children, we want to compare the indices
        // for the parent node of the branch nodes, thus we need to subtract 1 from our branch index.
        var aEndChildIndex = a.trail.indices[ i - 1 ];
        var bEndChildIndex = b.trail.indices[ i - 1 ];
        if ( aEndChildIndex < bEndChildIndex ) {
          return -1;
        }
        else if ( aEndChildIndex > bEndChildIndex ) {
          return 1;
        }
        else {
          throw new Error( 'Two different children have the same child index' );
        }
      } );

      var containerElement = this.peer.getChildContainerElement();
      for ( var n = this.children.length - 1; n >= 0; n-- ) {
        var peerDOMElement = this.children[ n ].peer.domElement;

        // if the peer has a parent container element, this structure containing the peerDOMElement should be inserted
        if ( this.children[ n ].peer.hasParentContainer() ) {
          peerDOMElement = this.children[ n ].peer.getParentContainerElement();
        }
        if ( peerDOMElement === containerElement.childNodes[ n ] ) {
          continue;
        }
        containerElement.insertBefore( peerDOMElement, containerElement.childNodes[ n + 1 ] );
      }
    },

    // Recursive disposal
    dispose: function() {
      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance(
        'Disposing ' + this.toString() );
      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.push();

      // Disconnect DOM and remove listeners
      if ( !this.isRootInstance ) {
        this.peer.dispose();

        // remove this peer's dom element (or its parent container) from the parent peer's
        // dom element (or its child container), disabling input so that we do not call input listeners
        // while the elements are removed from the DOM
        this.node.accessibleInputEnabled = false;
        this.parent.peer.getChildContainerElement().removeChild( this.peer.getParentContainerElement() );
        this.node.accessibleInputEnabled = true;

        // remove visibility/accessibleVisibility listeners that were added
        for ( var i = 0; i < this.trailDiff.length; i++ ) {
          this.trailDiff[ i ].offStatic( 'visibility', this.accessibleVisibilityListener );
          this.trailDiff[ i ].accessibleVisibilityChangedEmitter.removeListener( this.accessibleVisibilityListener );
        }
      }

      while ( this.children.length ) {
        this.children.pop().dispose();
      }

      // If we are the root accessible instance, we won't actually have a reference to a node.
      if ( this.node ) {
        this.node.removeAccessibleInstance( this );
      }

      this.display = null;
      this.trail = null;
      this.node = null;
      this.peer = null;
      this.disposed = true;

      this.freeToPool();

      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.pop();
    },

    toString: function() {
      return this.id + '#{' + this.trail.toString() + '}';
    },

    auditRoot: function() {
      assert && assert( this.trail.length === 0,
        'Should only call auditRoot() on the root AccessibleInstance for a display' );

      function audit( nestedOrderArray, accessibleInstance ) {
        assert && assert( nestedOrderArray.length === accessibleInstance.children.length,
          'Different number of children in accessible instance' );

        _.each( nestedOrderArray, function( nestedChild ) {
          var instance = _.find( accessibleInstance.children, function( childInstance ) {
            return childInstance.trail.equals( nestedChild.trail );
          } );
          assert && assert( instance, 'Missing child accessible instance' );

          audit( nestedChild.children, instance );
        } );

        // Exact Order checks
        for ( var i = 0; i < nestedOrderArray.length; i++ ) {
          assert && assert( nestedOrderArray[ i ].trail.lastNode() === accessibleInstance.children[ i ].node,
            'Accessible order mismatch' );
        }
      }

      audit( this.display.rootNode.getNestedAccessibleOrder(), this );
    }
  } );

  Poolable.mixInto( AccessibleInstance, {
    constructorDuplicateFactory: function( pool ) {
      return function( parent, display, trail ) {
        if ( pool.length ) {
          return pool.pop().initializeAccessibleInstance( parent, display, trail );
        }
        else {
          return new AccessibleInstance( parent, display, trail );
        }
      };
    }
  } );

  return AccessibleInstance;
} );
