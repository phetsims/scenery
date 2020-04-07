// Copyright 2015-2020, University of Colorado Boulder

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

import Events from '../../../../axon/js/Events.js';
import cleanArray from '../../../../phet-core/js/cleanArray.js';
import inherit from '../../../../phet-core/js/inherit.js';
import platform from '../../../../phet-core/js/platform.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import scenery from '../../scenery.js';
import TransformTracker from '../../util/TransformTracker.js';
import AccessibilityUtils from './AccessibilityUtils.js';
import AccessiblePeer from './AccessiblePeer.js';

let globalId = 1;

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

    assert && assert( !this.id || this.isDisposed, 'If we previously existed, we need to have been disposed' );

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

    // @private {number} - The number of nodes in our trail that are NOT in our parent's trail and do NOT have our
    // display in their accessibleDisplays. For non-root instances, this is initialized later in the constructor.
    this.invisibleCount = 0;

    // @private {Array.<Node>} - Nodes that are in our trail (but not those of our parent)
    this.relativeNodes = [];

    // @private {Array.<boolean>} - Whether our display is in the respective relativeNodes' accessibleDisplays
    this.relativeVisibilities = [];

    // @private {function} - The listeners added to the respective relativeNodes
    this.relativeListeners = [];

    // @public (scenery-internal) {TransformTracker|null} - Used to quickly compute the global matrix of this
    // instance's transform source Node and observe when the transform changes. Used by AccessiblePeer to update
    // positioning of sibling elements. By default, watches this AccessibleInstance's visual trail.
    this.transformTracker = null;
    this.updateTransformTracker( this.node ? this.node.pdomTransformSourceNode : null );

    // @private {boolean} - Whether we are currently in a "disposed" (in the pool) state, or are available to be
    // re-initialized
    this.isDisposed = false;

    if ( this.isRootInstance ) {
      const accessibilityContainer = document.createElement( 'div' );
      this.peer = AccessiblePeer.createFromPool( this, {
        primarySibling: accessibilityContainer
      } );
    }
    else {
      this.peer = AccessiblePeer.createFromPool( this );

      // The peer is not fully constructed until this update function is called, see https://github.com/phetsims/scenery/issues/832
      this.peer.update();

      assert && assert( this.peer.primarySibling, 'accessible peer must have a primarySibling upon completion of construction' );

      // Scan over all of the nodes in our trail (that are NOT in our parent's trail) to check for accessibleDisplays
      // so we can initialize our invisibleCount and add listeners.
      const parentTrail = this.parent.trail;
      for ( let i = parentTrail.length; i < trail.length; i++ ) {
        const relativeNode = trail.nodes[ i ];
        this.relativeNodes.push( relativeNode );

        const accessibleDisplays = relativeNode._accessibleDisplaysInfo.accessibleDisplays;
        const isVisible = _.includes( accessibleDisplays, display );
        this.relativeVisibilities.push( isVisible );
        if ( !isVisible ) {
          this.invisibleCount++;
        }

        const listener = this.checkAccessibleDisplayVisibility.bind( this, i - parentTrail.length );
        relativeNode.onStatic( 'accessibleDisplays', listener );
        this.relativeListeners.push( listener );
      }

      this.updateVisibility();
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
    sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance(
      'addConsecutiveInstances on ' + this.toString() + ' with: ' + accessibleInstances.map( function( inst ) { return inst.toString(); } ).join( ',' ) );
    sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.push();

    const hadChildren = this.children.length > 0;

    Array.prototype.push.apply( this.children, accessibleInstances );

    for ( let i = 0; i < accessibleInstances.length; i++ ) {
      // Append the container parent to the end (so that, when provided in order, we don't have to resort below
      // when initializing).
      AccessibilityUtils.insertElements( this.peer.primarySibling, accessibleInstances[ i ].peer.topLevelElements );
    }

    if ( hadChildren ) {
      this.sortChildren();
    }

    sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.pop();
  },

  /**
   * Removes any child instances that are based on the provided trail.
   * @public
   *
   * @param {Trail} trail
   */
  removeInstancesForTrail: function( trail ) {
    sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance(
      'removeInstancesForTrail on ' + this.toString() + ' with trail ' + trail.toString() );
    sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.push();

    for ( let i = 0; i < this.children.length; i++ ) {
      const childInstance = this.children[ i ];
      const childTrail = childInstance.trail;

      // Not worth it to inspect before our trail ends, since it should be (!) guaranteed to be equal
      let differs = childTrail.length < trail.length;
      if ( !differs ) {
        for ( let j = this.trail.length; j < trail.length; j++ ) {
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

    sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.pop();
  },

  /**
   * Removes all of the children.
   * @public
   */
  removeAllChildren: function() {
    sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.AccessibleInstance( 'removeAllChildren on ' + this.toString() );
    sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.push();

    while ( this.children.length ) {
      this.children.pop().dispose();
    }

    sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.pop();
  },

  /**
   * Returns an AccessibleInstance child (if one exists with the given Trail), or null otherwise.
   * @public
   *
   * @param {Trail} trail
   * @returns {AccessibleInstance|null}
   */
  findChildWithTrail: function( trail ) {
    for ( let i = 0; i < this.children.length; i++ ) {
      const child = this.children[ i ];
      if ( child.trail.equals( trail ) ) {
        return child;
      }
    }
    return null;
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

    for ( let i = this.children.length - 1; i >= 0; i-- ) {
      const childInstance = this.children[ i ];
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
   * Checks to see whether our visibility needs an update based on an accessibleDisplays change.
   * @private
   *
   * @param {number} index - Index into the relativeNodes array (which node had the notification)
   */
  checkAccessibleDisplayVisibility: function( index ) {
    const isNodeVisible = _.includes( this.relativeNodes[ index ]._accessibleDisplaysInfo.accessibleDisplays, this.display );
    const wasNodeVisible = this.relativeVisibilities[ index ];

    if ( isNodeVisible !== wasNodeVisible ) {
      this.relativeVisibilities[ index ] = isNodeVisible;

      const wasVisible = this.invisibleCount === 0;

      this.invisibleCount += ( isNodeVisible ? -1 : 1 );
      assert && assert( this.invisibleCount >= 0 && this.invisibleCount <= this.relativeNodes.length );

      const isVisible = this.invisibleCount === 0;

      if ( isVisible !== wasVisible ) {
        this.updateVisibility();
      }
    }
  },

  /**
   * Update visibility of this peer's accessible DOM content. The hidden attribute will hide all of the descendant
   * DOM content, so it is not necessary to update the subtree of AccessibleInstances since the browser
   * will do this for us.
   * @private
   */
  updateVisibility: function() {
    this.peer.setVisible( this.invisibleCount <= 0 );

    // if we hid a parent element, blur focus if active element was an ancestor
    if ( !this.peer.isVisible() ) {
      if ( this.peer.primarySibling.contains( document.activeElement ) ) { // still true if activeElement is this primary sibling
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
   * Returns whether the parallel DOM for this instance and its ancestors are not hidden.
   * @public
   *
   * @returns {boolean}
   */
  isGloballyVisible: function() {

    // If this peer is hidden, then return because that attribute will bubble down to children,
    // otherwise recurse to parent.
    if ( !this.peer.isVisible() ) {
      return false;
    }
    else if ( this.parent ) {
      return this.parent.isGloballyVisible();
    }
    else { // base case at root
      return true;
    }
  },

  /**
   * Returns what our list of children (after sorting) should be.
   * @private
   *
   * @param {Trail} trail - A partial trail, where the root of the trail is either this.node or the display's root
   *                        node (if we are the root AccessibleInstance)
   * @returns {Array.<AccessibleInstance>}
   */
  getChildOrdering: function( trail ) {
    const node = trail.lastNode();
    const effectiveChildren = node.getEffectiveChildren();
    let i;
    const instances = [];

    // base case, node has accessible content, but don't match the "root" node of this accessible instance
    if ( node.accessibleContent && node !== this.node ) {
      const potentialInstances = node.accessibleInstances;

      instanceLoop:
        for ( i = 0; i < potentialInstances.length; i++ ) {
          const potentialInstance = potentialInstances[ i ];
          if ( potentialInstance.parent !== this ) {
            continue instanceLoop;
          }

          for ( let j = 0; j < trail.length; j++ ) {
            if ( trail.nodes[ j ] !== potentialInstance.trail.nodes[ j + potentialInstance.trail.length - trail.length ] ) {
              continue instanceLoop;
            }
          }

          instances.push( potentialInstance ); // length will always be 1
        }

      assert && assert( instances.length <= 1, 'If we select more than one this way, we have problems' );
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
    // It's simpler/faster to just grab our order directly with one recursion, rather than specifying a sorting
    // function (since a lot gets re-evaluated in that case).
    const targetChildren = this.getChildOrdering( new scenery.Trail( this.isRootInstance ? this.display.rootNode : this.node ) );

    assert && assert( targetChildren.length === this.children.length, 'sorting should not change number of children' );

    // {Array.<AccessibleInstance>}
    this.children = targetChildren;

    // the DOMElement to add the child DOMElements to.
    const primarySibling = this.peer.primarySibling;

    // "i" will keep track of the "collapsed" index when all DOMElements for all AccessibleInstance children are
    // added to a single parent DOMElement (this AccessibleInstance's AccessiblePeer's primarySibling)
    let i = primarySibling.childNodes.length - 1;

    // Iterate through all AccessibleInstance children
    for ( let peerIndex = this.children.length - 1; peerIndex >= 0; peerIndex-- ) {
      const peer = this.children[ peerIndex ].peer;

      // Iterate through all top level elements of an AccessibleInstance's peer
      for ( let elementIndex = peer.topLevelElements.length - 1; elementIndex >= 0; elementIndex-- ) {
        const element = peer.topLevelElements[ elementIndex ];

        // Reorder DOM elements in a way that doesn't do any work if they are already in a sorted order.
        // No need to reinsert if `element` is already in the right order
        if ( primarySibling.childNodes[ i ] !== element ) {
          primarySibling.insertBefore( element, primarySibling.childNodes[ i + 1 ] );
        }

        // Decrement so that it is easier to place elements using the browser's Node.insertBefore api
        i--;
      }
    }
  },

  /**
   * Create a new TransformTracker that will observe transforms along the trail of this AccessibleInstance OR
   * the specified pdomTransformSourceNode. See ParallelPDOM.setPDOMTransformSourceNode(). The source Node
   * must not use DAG so that its trail is unique.
   * @public
   *
   * @param {Node|null} node - if null, we use this AccessibleInstance's visual trail
   * @returns {TransformTracker}
   */
  updateTransformTracker( pdomTransformSourceNode ) {
    this.transformTracker && this.transformTracker.dispose();

    let trackedTrail = null;
    if ( pdomTransformSourceNode ) {
      trackedTrail = pdomTransformSourceNode.getUniqueTrail();
    }
    else {
      trackedTrail = AccessibleInstance.guessVisualTrail( this.trail, this.display.rootNode );
    }

    this.transformTracker = new TransformTracker( trackedTrail );
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

      // remove this peer's primary sibling DOM Element (or its container parent) from the parent peer's
      // primary sibling (or its child container)
      AccessibilityUtils.removeElements( this.parent.peer.primarySibling, this.peer.topLevelElements );

      for ( let i = 0; i < this.relativeNodes.length; i++ ) {
        this.relativeNodes[ i ].offStatic( 'accessibleDisplays', this.relativeListeners[ i ] );
      }
    }

    while ( this.children.length ) {
      this.children.pop().dispose();
    }

    // NOTE: We dispose OUR peer after disposing children, so our peer can be available for our children during
    // disposal.
    this.peer.dispose();

    // dispose after the peer so the peer can remove any listeners from it
    this.transformTracker.dispose();
    this.transformTracker = null;

    // If we are the root accessible instance, we won't actually have a reference to a node.
    if ( this.node ) {
      this.node.removeAccessibleInstance( this );
    }

    this.relativeNodes = null;
    this.display = null;
    this.trail = null;
    this.node = null;
    this.peer = null;
    this.isDisposed = true;

    this.freeToPool();

    sceneryLog && sceneryLog.AccessibleInstance && sceneryLog.pop();
  },

  /**
   * For debugging purposes.
   * @public
   *
   * @returns {string}
   */
  toString: function() {
    return this.id + '#{' + this.trail.toString() + '}';
  },

  /**
   * For debugging purposes, inspect the tree of AccessibleInstances from the root.
   *
   * Only ever called from the _rootAccessibleInstance of the display.
   *
   * @public (scenery-internal)
   */
  auditRoot: function() {
    if ( !assert ) { return; }

    const rootNode = this.display.rootNode;

    assert( this.trail.length === 0,
      'Should only call auditRoot() on the root AccessibleInstance for a display' );

    function audit( fakeInstance, accessibleInstance ) {
      assert( fakeInstance.children.length === accessibleInstance.children.length,
        'Different number of children in accessible instance' );

      assert( fakeInstance.node === accessibleInstance.node, 'Node mismatch for AccessibleInstance' );

      for ( var i = 0; i < accessibleInstance.children.length; i++ ) {
        audit( fakeInstance.children[ i ], accessibleInstance.children[ i ] );
      }

      const isVisible = accessibleInstance.isGloballyVisible();

      let shouldBeVisible = true;
      for ( i = 0; i < accessibleInstance.trail.length; i++ ) {
        const node = accessibleInstance.trail.nodes[ i ];
        const trails = node.getTrailsTo( rootNode ).filter( function( trail ) {
          return trail.isAccessibleVisible();
        } );
        if ( trails.length === 0 ) {
          shouldBeVisible = false;
          break;
        }
      }

      assert( isVisible === shouldBeVisible, 'Instance visibility mismatch' );
    }

    audit( AccessibleInstance.createFakeAccessibleTree( rootNode ), this );
  }
}, {

  /**
   * Since a "Trail" on AccessibleInstance can have discontinuous jumps (due to accessibleOrder), this finds the best
   * actual visual Trail to use, from the trail of an AccessibleInstance to the root of a Display.
   * @public
   *
   * @param {Trail} trail - trail of the AccessibleInstance, which can containe "gaps"
   * @param {Node} rootNode - root of a Display
   * @returns {Trail}
   */
  guessVisualTrail: function( trail, rootNode ) {
    trail.reindex();

    // Search for places in the trail where adjacent nodes do NOT have a parent-child relationship, i.e.
    // !nodes[ n ].hasChild( nodes[ n + 1 ] ).
    // NOTE: This index points to the parent where this is the case, because the indices in the trail are such that:
    // trail.nodes[ n ].children[ trail.indices[ n ] ] = trail.nodes[ n + 1 ]
    const lastBadIndex = trail.indices.lastIndexOf( -1 );

    // If we have no bad indices, just return our trail immediately.
    if ( lastBadIndex < 0 ) {
      return trail;
    }

    const firstGoodIndex = lastBadIndex + 1;
    const firstGoodNode = trail.nodes[ firstGoodIndex ];
    const baseTrails = firstGoodNode.getTrailsTo( rootNode );

    // firstGoodNode might not be attached to a Display either! Maybe client just hasn't gotten to it yet, so we
    // fail gracefully-ish?
    // assert && assert( baseTrails.length > 0, '"good node" in trail with gap not attached to root')
    if ( baseTrails.length === 0 ) {
      return trail;
    }

    // Add the rest of the trail back in
    const baseTrail = baseTrails[ 0 ];
    for ( let i = firstGoodIndex + 1; i < trail.length; i++ ) {
      baseTrail.addDescendant( trail.nodes[ i ] );
    }

    assert && assert( baseTrail.isValid(), 'trail not valid: ' + trail.uniqueId );

    return baseTrail;
  },

  /**
   * Creates a fake AccessibleInstance-like tree structure (with the equivalent nodes and children structure).
   * For debugging.
   * @private
   *
   * @param {Node} rootNode
   * @returns {Object} - Type FakeAccessibleInstance: { node: {Node}, children: {Array.<FakeAccessibleInstance>} }
   */
  createFakeAccessibleTree: function( rootNode ) {
    function createFakeTree( node ) {
      let fakeInstances = _.flatten( node.getEffectiveChildren().map( createFakeTree ) );
      if ( node.accessibleContent ) {
        fakeInstances = [ {
          node: node,
          children: fakeInstances
        } ];
      }
      return fakeInstances;
    }

    return {
      node: null,
      children: createFakeTree( rootNode )
    };
  }
} );

Poolable.mixInto( AccessibleInstance, {
  initialize: AccessibleInstance.prototype.initializeAccessibleInstance
} );

export default AccessibleInstance;