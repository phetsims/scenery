// Copyright 2018, University of Colorado Boulder

/**
 * Per-node information required to track what accessible Displays our node is visible under. Acts like a multimap
 * (duplicates allowed) to indicate how many times we appear in an accessible display.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * Tracks accessible display information for our given node.
   * @public (scenery-internal)
   * @constructor
   *
   * @param {Node} node
   */
  function AccessibleDisplaysInfo( node ) {

    // @public {Node}
    this.node = node;

    // @public (scenery-internal) {Array.<Display>} - (duplicates allowed) - There is one copy of each accessible
    // Display for each trail (from its root node to this node) that is fully visible (assuming this subtree is
    // accessible).
    // Thus, the value of this is:
    // - If this node is invisible OR the subtree has no accessibleContent/accessibleOrder: []
    // - Otherwise, it is the concatenation of our parents' accessibleDisplays (AND any accessible displays rooted
    //   at this node).
    // This value is synchronously updated, and supports accessibleInstances by letting them know when certain
    // nodes are visible on the display.
    this.accessibleDisplays = [];
  }

  scenery.register( 'AccessibleDisplaysInfo', AccessibleDisplaysInfo );

  return inherit( Object, AccessibleDisplaysInfo, {
    /**
     * Called when the node is added as a child to this node AND the node's subtree contains accessible content.
     * @public (scenery-internal)
     *
     * @param {Node} node
     */
    onAddChild: function( node ) {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'onAddChild n#' + node.id + ' (parent:n#' + this.node.id + ')' );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      if ( node._accessibleDisplaysInfo.canHaveAccessibleDisplays() ) {
        node._accessibleDisplaysInfo.addAccessibleDisplays( this.accessibleDisplays );
      }

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    },

    /**
     * Called when the node is removed as a child from this node AND the node's subtree contains accessible content.
     * @public (scenery-internal)
     *
     * @param {Node} node
     */
    onRemoveChild: function( node ) {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'onRemoveChild n#' + node.id + ' (parent:n#' + this.node.id + ')' );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      if ( node._accessibleDisplaysInfo.canHaveAccessibleDisplays() ) {
        node._accessibleDisplaysInfo.removeAccessibleDisplays( this.accessibleDisplays );
      }

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    },

    /**
     * Called when our summary bitmask changes
     * @public (scenery-internal)
     *
     * @param {number} oldBitmask
     * @param {number} newBitmask
     */
    onSummaryChange: function( oldBitmask, newBitmask ) {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'onSummaryChange n#' + this.node.id + ' wasA11y:' + !( Renderer.bitmaskNotAccessible & oldBitmask ) + ', isA11y:' + !( Renderer.bitmaskNotAccessible & newBitmask ) );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      // If we are invisible, our accessibleDisplays would not have changed ([] => [])
      if ( this.node.visible && this.node.accessibleVisible ) {
        var wasAccessible = !( Renderer.bitmaskNotAccessible & oldBitmask );
        var isAccessible = !( Renderer.bitmaskNotAccessible & newBitmask );

        // If we changed to be accessible, we need to recursively add accessible displays.
        if ( isAccessible && !wasAccessible ) {
          this.addAllAccessibleDisplays();
        }

        // If we changed to NOT be accessible, we need to recursively remove accessible displays.
        if ( !isAccessible && wasAccessible ) {
          this.removeAllAccessibleDisplays();
        }
      }

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    },

    /**
     * Called when our visibility changes.
     * @public (scenery-internal)
     *
     * @param {boolean} visible
     */
    onVisibilityChange: function( visible ) {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'onVisibilityChange n#' + this.node.id + ' visible:' + visible );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      // If we are not accessible (or accessibleVisible), our accessibleDisplays would not have changed ([] => [])
      if ( this.node.accessibleVisible && !this.node._rendererSummary.isNotAccessible() ) {
        if ( visible ) {
          this.addAllAccessibleDisplays();
        }
        else {
          this.removeAllAccessibleDisplays();
        }
      }

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    },

    /**
     * Called when our accessibleVisibility changes.
     * @public (scenery-internal)
     *
     * @param {boolean} visible
     */
    onAccessibleVisibilityChange: function( visible ) {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'onAccessibleVisibilityChange n#' + this.node.id + ' accessibleVisible:' + visible );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      // If we are not accessible, our accessibleDisplays would not have changed ([] => [])
      if ( this.node.visible && !this.node._rendererSummary.isNotAccessible() ) {
        if ( visible ) {
          this.addAllAccessibleDisplays();
        }
        else {
          this.removeAllAccessibleDisplays();
        }
      }

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    },

    /**
     * Called when we have a rooted display added to this node.
     * @public (scenery-internal)
     *
     * @param {Display} display
     */
    onAddedRootedDisplay: function( display ) {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'onAddedRootedDisplay n#' + this.node.id );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      if ( display._accessible && this.canHaveAccessibleDisplays() ) {
        this.addAccessibleDisplays( [ display ] );
      }

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    },

    /**
     * Called when we have a rooted display removed from this node.
     * @public (scenery-internal)
     *
     * @param {Display} display
     */
    onRemovedRootedDisplay: function( display ) {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'onRemovedRootedDisplay n#' + this.node.id );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      if ( display._accessible && this.canHaveAccessibleDisplays() ) {
        this.removeAccessibleDisplays( [ display ] );
      }

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    },

    /**
     * Returns whether we can have accessibleDisplays specified in our array.
     * @public (scenery-internal)
     *
     * @returns {boolean}
     */
    canHaveAccessibleDisplays: function() {
      return this.node.visible && this.node.accessibleVisible && !this.node._rendererSummary.isNotAccessible();
    },

    /**
     * Adds all of our accessible displays to our array (and propagates).
     * @private
     */
    addAllAccessibleDisplays: function() {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'addAllAccessibleDisplays n#' + this.node.id );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      assert && assert( this.accessibleDisplays.length === 0, 'Should be empty before adding everything' );
      assert && assert( this.canHaveAccessibleDisplays(), 'Should happen when we can store accessibleDisplays' );

      var i;
      var displays = [];

      // Concatenation of our parents' accessibleDisplays
      for ( i = 0; i < this.node._parents.length; i++ ) {
        Array.prototype.push.apply( displays, this.node._parents[ i ]._accessibleDisplaysInfo.accessibleDisplays );
      }

      // AND any acessible displays rooted at this node
      for ( i = 0; i < this.node._rootedDisplays.length; i++ ) {
        var display = this.node._rootedDisplays[ i ];
        if ( display._accessible ) {
          displays.push( display );
        }
      }

      this.addAccessibleDisplays( displays );

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    },

    /**
     * Removes all of our accessible displays from our array (and propagates).
     * @private
     */
    removeAllAccessibleDisplays: function() {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'removeAllAccessibleDisplays n#' + this.node.id );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      assert && assert( !this.canHaveAccessibleDisplays(), 'Should happen when we cannot store accessibleDisplays' );

      // TODO: is there a way to avoid a copy?
      this.removeAccessibleDisplays( this.accessibleDisplays.slice() );

      assert && assert( this.accessibleDisplays.length === 0, 'Should be empty after removing everything' );

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    },

    /**
     * Adds a list of accessible displays to our internal list. See accessibleDisplays documentation.
     * @private
     *
     * @param {Array.<Display>} displays
     */
    addAccessibleDisplays: function( displays ) {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'addAccessibleDisplays n#' + this.node.id + ' numDisplays:' + displays.length );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      assert && assert( Array.isArray( displays ) );

      // Simplifies things if we can stop no-ops here.
      if ( displays.length !== 0 ) {
        Array.prototype.push.apply( this.accessibleDisplays, displays );

        // Propagate the change to our children
        for ( var i = 0; i < this.node._children.length; i++ ) {
          var child = this.node._children[ i ];
          if ( child._accessibleDisplaysInfo.canHaveAccessibleDisplays() ) {
            this.node._children[ i ]._accessibleDisplaysInfo.addAccessibleDisplays( displays );
          }
        }

        this.node.trigger0( 'accessibleDisplays' );
      }

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    },

    /**
     * Removes a list of accessible displays from our internal list. See accessibleDisplays documentation.
     * @private
     *
     * @param {Array.<Display>} displays
     */
    removeAccessibleDisplays: function( displays ) {
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.AccessibleDisplaysInfo( 'removeAccessibleDisplays n#' + this.node.id + ' numDisplays:' + displays.length );
      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.push();

      assert && assert( Array.isArray( displays ) );
      assert && assert( this.accessibleDisplays.length >= displays.length );

      // Simplifies things if we can stop no-ops here.
      if ( displays.length !== 0 ) {
        var i;

        for ( i = displays.length - 1; i >= 0; i-- ) {
          var index = this.accessibleDisplays.lastIndexOf( displays[ i ] );
          assert && assert( index >= 0 );
          this.accessibleDisplays.splice( i, 1 );
        }

        // Propagate the change to our children
        for ( i = 0; i < this.node._children.length; i++ ) {
          var child = this.node._children[ i ];
          // NOTE: Since this gets called many times from the RendererSummary (which happens before the actual child
          // modification happens), we DO NOT want to traverse to the child node getting removed. Ideally a better
          // solution than this flag should be found.
          if ( child._accessibleDisplaysInfo.canHaveAccessibleDisplays() && !child._isGettingRemovedFromParent ) {
            child._accessibleDisplaysInfo.removeAccessibleDisplays( displays );
          }
        }

        this.node.trigger0( 'accessibleDisplays' );
      }

      sceneryLog && sceneryLog.AccessibleDisplaysInfo && sceneryLog.pop();
    }
  } );
} );
