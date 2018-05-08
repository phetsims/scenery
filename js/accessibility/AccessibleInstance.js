// Copyright 2015-2016, University of Colorado Boulder

/**
 * An instance that is synchronously created, for handling accessibility needs.
 *
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
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var AccessiblePeer = require( 'SCENERY/accessibility/AccessiblePeer' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Events = require( 'AXON/Events' );
  var inherit = require( 'PHET_CORE/inherit' );
  var KeyboardUtil = require( 'SCENERY/accessibility/KeyboardUtil' );
  var platform = require( 'PHET_CORE/platform' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );

  var globalId = 1;

  /**
   * Constructor for AccessibleInstance, uses an initialize method for pooling.
   * @constructor
   * @mixes Poolable
   *
   * @param {AccessibleInstance|null} parent - parent of this instance, null if root of AccessibleInstance tree
   * @param {Display} display
   * @param {Trail} trail - trail to the node for this AccessibleInstance
   */
  function AccessibleInstance( parent, display, trail ) {
    this.initializeAccessibleInstance( parent, display, trail );
  }

  scenery.register( 'AccessibleInstance', AccessibleInstance );

  inherit( Events, AccessibleInstance, {

    /**
     * Initializes an AccessibleInstance, implements construction for pooling.
     * @private
     *
     * @param {AccessibleInstance|null} parent - null if this AccessibleInstance is root of AccessibleInstance tree
     * @param {Display} display
     * @param {Trail} trail - trail to node for this AccessibleInstance
     * @returns {AccessibleInstance} - Returns 'this' reference, for chaining
     */
    initializeAccessibleInstance: function( parent, display, trail ) {
      Events.call( this ); // TODO: is Events worth mixing in by default? Will we need to listen to events?

      assert && assert( !this.id || this.disposed, 'If we previously existed, we need to have been disposed' );

      // unique ID
      this.id = this.id || globalId++;

      this.parent = parent;

      // @public {Display}
      this.display = display;

      // @public {Trail}
      this.trail = trail;

      // @public {boolean}
      this.isRootInstance = parent === null;

      // @public {Node|null}
      this.node = this.isRootInstance ? null : trail.lastNode();

      // @public {Array.<AccessibleInstance>}
      this.children = cleanArray( this.children );

      // If we are the root accessible instance, we won't actually have a reference to a node.
      if ( this.node ) {
        this.node.addAccessibleInstance( this );
      }

      // @public {AccessiblePeer}
      this.peer = null; // Filled in below

      if ( this.isRootInstance ) {
        var accessibilityContainer = document.createElement( 'div' );

        // give the container a class name so it is hidden in the Display, see accessibility styling in Display.js
        accessibilityContainer.className = 'accessibility';
        this.peer = new AccessiblePeer( this, accessibilityContainer );

        var self = this;
        document.body.addEventListener( 'keydown', function( event ) {

          // if an accessible node was being interacted with a mouse, or had focus when sim is made inactive, this node
          // should receive focus upon resuming keyboard navigation
          if ( self.display.pointerFocus || self.display.activeNode ) {
            var active = self.display.pointerFocus || self.display.activeNode;
            var focusable = active.focusable;

            // if there is a single accessible instance, we can restore focus
            if ( active.getAccessibleInstances().length === 1 ) {

              // if all ancestors of this node are visible, so is the active node
              var nodeAndAncestorsVisible = true;
              var activeTrail = active.accessibleInstances[ 0 ].trail;
              for ( var i = activeTrail.nodes.length - 1; i >= 0; i-- ) {
                if ( !activeTrail.nodes[ i ].visible ) {
                  nodeAndAncestorsVisible = false;
                  break;
                }
              }

              if ( focusable && nodeAndAncestorsVisible ) {
                if ( event.keyCode === KeyboardUtil.KEY_TAB ) {
                  event.preventDefault();
                  active.focus();
                  self.display.pointerFocus = null;
                  self.display.activeNode = null;
                }
              }
            }
          }
        } );
      }
      else {
        this.peer = this.node.accessibleContent.createPeer( this );
        var childContainerElement = this.parent.peer.getChildContainerElement();

        // insert the peer's DOM element or its parent if it is contained in a parent element for structure
        childContainerElement.insertBefore( this.peer.getContainerParent(), childContainerElement.childNodes[ 0 ] );

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
     * Adds a series of (sorted) accessible instances as children.
     * @public
     *
     * @param {Array.<AccessibleInstance>} accessibleInstances
     */
    addConsecutiveInstances: function( accessibleInstances ) {
      var hadChildren = this.children.length > 0;

      Array.prototype.push.apply( this.children, accessibleInstances );

      if ( hadChildren ) {
        this.sortChildren();
      }
    },

    /**
     * Removes any child instances that are based on the provided trail.
     * @public
     *
     * @param {Trail} trail
     */
    removeInstancesForTrail: function( trail ) {
      for ( var i = 0; i < this.children.length; i++ ) {
        var childInstance = this.children[ i ];
        var childTrail = childInstance.trail;

        // Not worth it to inspect before our trail ends, since it should be (!) guaranteed to be equal
        var differs = childTrail.length < trail.length;
        if ( !differs ) {
          for ( var j = this.trail.length; j < trail.length; j++ ) {
            if ( trail.nodes[ j ] !== childTrail.nodes[ j ] ) {
              differs = true;
              break;
            }
          }
        }

        if ( !differs ) {
          this.children.splice( i, 1 );
          childInstance.dispose();
          i -= 1;
        }
      }
    },

    /**
     * Remove a subtree of AccessibleInstances from this AccessibleInstance
     *
     * @param {Trail} trail - children of this AccessibleInstance will be removed if the child trails are extensions
     *                        of the trail.
     * @public (scenery-internal)
     */
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

      var parentElement = this.peer.getContainerParent();

      var self = this;
      parentElement.hidden = !( visibilityCount === self.trailDiff.length );

      // if we hid a parent element, blur focus if active element was an ancestor
      if ( parentElement.hidden ) {
        if ( parentElement.contains( document.activeElement ) ) {
          scenery.Display.focus = null;
        }
      }

      // Edge has a bug where removing the hidden attribute on an ancestor doesn't add elements back to the navigation
      // order. As a workaround, forcing the browser to redraw the PDOM seems to fix the issue. Forced redraw method
      // recommended by https://stackoverflow.com/questions/8840580/force-dom-redraw-refresh-on-chrome-mac, also see
      // https://github.com/phetsims/a11y-research/issues/30
      if ( platform.edge ) {
        this.display.getAccessibleDOMElement().style.display = 'none';
        this.display.getAccessibleDOMElement().style.display = 'block';
      }
    },

    /**
     * Returns what our list of children (after sorting) should be.
     * @private
     *
     * @param {Trail} trail - A partial trail, "ending" (with its root) at this.node.
     * @returns {Array.<AccessibleInstance>}
     */
    getChildOrdering: function( trail ) {
      var node = trail.lastNode();
      var effectiveChildren = node.getEffectiveChildren();
      var i;
      var instances = [];

      if ( node.accessibleContent && node !== this.node ) {
        var potentialInstances = node.accessibleInstances;

        instanceLoop:
        for ( i = 0; i < potentialInstances.length; i++ ) {
          var potentialInstance = potentialInstances[ i ];
          if ( potentialInstance.parent !== this ) {
            continue instanceLoop;
          }

          for ( var j = 0; j < trail.length; j++ ) {
            if ( trail.nodes[ j ] !== potentialInstance.trail.nodes[ j + potentialInstance.trail.length - trail.length ] ) {
              continue instanceLoop;
            }
          }

          instances.push( potentialInstance );
        }

        assert && assert( instances.length >= 1, 'If we select more than one this way, we have problems' );
      }
      else {
        for ( i = 0; i < effectiveChildren.length; i++ ) {
          trail.addDescendant( effectiveChildren[ i ], i );
          Array.prototype.push.apply( instances, this.getChildOrdering( trail ) );
          trail.removeDescendant();
        }
      }

      return instances;
    },

    /**
     * Sort our child accessible instances in the order they should appear in the parallel DOM. We do this by
     * creating a comparison function between two accessible instances. The function walks along the trails
     * of the children, looking for specified accessible orders that would determine the ordering for the two
     * AccessibleInstances.
     *
     * @public (scenery-internal)
     */
    sortChildren: function() {
      var targetChildren = this.getChildOrdering( new scenery.Trail( this.node ) );

      assert && assert( targetChildren.length === this.children.length );
      this.children = targetChildren;

      var containerElement = this.peer.getChildContainerElement();
      for ( var n = this.children.length - 1; n >= 0; n-- ) {
        var peerDOMElement = this.children[ n ].peer.primarySibling;

        // if the peer has a container parent, this structure containing the peerDOMElement should be inserted
        if ( this.children[ n ].peer.hasContainerParent() ) {
          peerDOMElement = this.children[ n ].peer.getContainerParent();
        }
        if ( peerDOMElement === containerElement.childNodes[ n ] ) {
          continue;
        }
        containerElement.insertBefore( peerDOMElement, containerElement.childNodes[ n + 1 ] );
      }
    },

    /**
     * Recursive disposal, to make eligible for garbage collection.
     *
     * @public (scenery-internal)
     */
    dispose: function() {
      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance(
        'Disposing ' + this.toString() );
      sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.push();

      // Disconnect DOM and remove listeners
      if ( !this.isRootInstance ) {
        this.peer.dispose();

        // remove this peer's primary sibling DOM Element (or its container parent) from the parent peer's
        // primary sibling (or its child container)
        this.parent.peer.getChildContainerElement().removeChild( this.peer.getContainerParent() );

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

    /**
     * For debugging purposes.
     * @public
     *
     * @return {string}
     */
    toString: function() {
      return this.id + '#{' + this.trail.toString() + '}';
    },

    /**
     * For debugging purposes, inspect the tree of AccessibleInstances from the root.
     *
     * @public (scenery-internal)
     */
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
