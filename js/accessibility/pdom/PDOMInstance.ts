// Copyright 2015-2025, University of Colorado Boulder

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

import dotRandom from '../../../../dot/js/dotRandom.js';
import cleanArray from '../../../../phet-core/js/cleanArray.js';
import Pool from '../../../../phet-core/js/Pool.js';
import FocusManager from '../../accessibility/FocusManager.js';
import PDOMPeer from '../../accessibility/pdom/PDOMPeer.js';
import PDOMUtils from '../../accessibility/pdom/PDOMUtils.js';
import type Display from '../../display/Display.js';
import type Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';
import Trail from '../../util/Trail.js';
import TransformTracker from '../../util/TransformTracker.js';
import { getPDOMFocusedNode } from '../pdomFocusProperty.js';
import { guessVisualTrail } from './guessVisualTrail.js';
import { PDOM_UNIQUE_ID_SEPARATOR } from './PDOM_UNIQUE_ID_SEPARATOR.js';
import PDOMUniqueIdStrategy from './PDOMUniqueIdStrategy.js';
import UNIQUE_ID_STRATEGY from './UNIQUE_ID_STRATEGY.js';

// A type representing a fake instance, for some aggressive auditing (under ?assertslow)
type FakeInstance = {
  node: Node | null;
  children: FakeInstance[];
};

let globalId = 1;

class PDOMInstance {

  // unique ID
  private id!: number;

  public parent!: PDOMInstance | null;

  // (scenery-internal)
  public display!: Display | null;

  public trail!: Trail | null;
  public isRootInstance!: boolean;
  public node!: Node | null;
  public children!: PDOMInstance[];
  public peer!: PDOMPeer | null;
  public parentHeadingLevel!: number; // the parent/ancestor heading level (logically)
  public pendingHeadingLevel!: number; // heading level we expect to assign to the instance, if there is an accessibleHeading for the peer
  public headingLevel!: number | null; // the heading level used for a heading for this instance (or null)

  // {number} - The number of nodes in our trail that are NOT in our parent's trail and do NOT have our
  // display in their pdomDisplays. For non-root instances, this is initialized later in the constructor.
  private invisibleCount!: number;

  // {Array.<Node>} - Nodes that are in our trail (but not those of our parent)
  private relativeNodes: Node[] | null = [];

  // {Array.<boolean>} - Whether our display is in the respective relativeNodes' pdomDisplays
  private relativeVisibilities: boolean[] = [];

  // {function} - The listeners added to the respective relativeNodes
  private relativeListeners: ( () => void )[] = [];

  // (scenery-internal) {TransformTracker|null} - Used to quickly compute the global matrix of this
  // instance's transform source Node and observe when the transform changes. Used by PDOMPeer to update
  // positioning of sibling elements. By default, watches this PDOMInstance's visual trail.
  public transformTracker: TransformTracker | null = null;

  // {boolean} - Whether we are currently in a "disposed" (in the pool) state, or are available to be
  // re-initialized
  private isDisposed!: boolean;

  /**
   * Constructor for PDOMInstance, uses an initialize method for pooling.
   *
   * @param parent - parent of this instance, null if root of PDOMInstance tree
   * @param display
   * @param trail - trail to the node for this PDOMInstance
   */
  public constructor( parent: PDOMInstance | null, display: Display, trail: Trail, rootParentHeadingLevel?: number ) {
    this.initializePDOMInstance( parent, display, trail, rootParentHeadingLevel );
  }

  /**
   * Initializes a PDOMInstance, implements construction for pooling.
   *
   * @param parent - null if this PDOMInstance is root of PDOMInstance tree
   * @param display
   * @param trail - trail to node for this PDOMInstance
   * @returns - Returns 'this' reference, for chaining
   */
  public initializePDOMInstance( parent: PDOMInstance | null, display: Display, trail: Trail, rootParentHeadingLevel?: number ): PDOMInstance {
    assert && assert( !this.id || this.isDisposed, 'If we previously existed, we need to have been disposed' );

    // unique ID
    this.id = this.id || globalId++;

    this.parent = parent;

    // {Display}
    this.display = display;

    // {Trail}
    this.trail = trail;

    // {boolean}
    this.isRootInstance = parent === null;

    // {Node|null}
    this.node = this.isRootInstance ? null : trail.lastNode();

    // {Array.<PDOMInstance>}
    this.children = cleanArray( this.children );

    // If we are the root accessible instance, we won't actually have a reference to a node.
    if ( this.node ) {
      this.node.addPDOMInstance( this );
    }

    // {number} - The number of nodes in our trail that are NOT in our parent's trail and do NOT have our
    // display in their pdomDisplays. For non-root instances, this is initialized later in the constructor.
    this.invisibleCount = 0;

    // {Array.<Node>} - Nodes that are in our trail (but not those of our parent)
    this.relativeNodes = [];

    // {Array.<boolean>} - Whether our display is in the respective relativeNodes' pdomDisplays
    this.relativeVisibilities = [];

    // {function} - The listeners added to the respective relativeNodes
    this.relativeListeners = [];

    // (scenery-internal) {TransformTracker|null} - Used to quickly compute the global matrix of this
    // instance's transform source Node and observe when the transform changes. Used by PDOMPeer to update
    // positioning of sibling elements. By default, watches this PDOMInstance's visual trail.
    this.transformTracker = null;
    this.updateTransformTracker( this.node ? this.node.pdomTransformSourceNode : null );

    // {boolean} - Whether we are currently in a "disposed" (in the pool) state, or are available to be
    // re-initialized
    this.isDisposed = false;

    assert && assert( this.parent || rootParentHeadingLevel !== undefined, 'rootParentHeadingLevel required for root instance' );

    // NOTE: we are relying on `onPDOMContentChange()` to rebuild the PDOMInstance tree when the presence of an
    // accessibleHeading OR the value of accessibleHeadingIncrement changes! Thus these computed heading levels will
    // not change once constructed.
    this.parentHeadingLevel = this.parent ? ( this.parent.headingLevel ?? this.parent.parentHeadingLevel ) : rootParentHeadingLevel!;

    // An accessibleHeading may be computed by PDOMPeer.update, so wait to assign a headingLevel for descendants until
    // after the update call.
    this.pendingHeadingLevel = this.parentHeadingLevel + ( this.node ? this.node.accessibleHeadingIncrement : 1 );

    if ( this.isRootInstance ) {
      const accessibilityContainer = document.createElement( 'div' );

      // @ts-expect-error - Poolable is a mixin and TypeScript doesn't have good mixin support
      this.peer = PDOMPeer.createFromPool( this, {
        primarySibling: accessibilityContainer
      } );

      // The root level instance has no Node and therefore no heading level.
      this.headingLevel = null;
    }
    else {

      // @ts-expect-error - Poolable a mixin and TypeScript doesn't have good mixin support
      this.peer = PDOMPeer.createFromPool( this );

      // The peer is not fully constructed until this update function is called, see https://github.com/phetsims/scenery/issues/832
      // Trail Ids will never change, so update them eagerly, a single time during construction.
      this.peer!.update( UNIQUE_ID_STRATEGY === PDOMUniqueIdStrategy.TRAIL_ID );
      assert && assert( this.peer!.getPlaceableSibling(), 'accessible peer must have elements ready upon completion of construction' );

      // If the peer has a heading level after its update, then we need to set the heading level of this instance.
      this.headingLevel = this.peer!.headingSibling ? this.pendingHeadingLevel : null;
      assert && assert( this.headingLevel === null || ( this.headingLevel >= 1 && this.headingLevel <= 6 ), `Heading level of h${this.headingLevel} is invalid` );

      // Scan over all of the nodes in our trail (that are NOT in our parent's trail) to check for pdomDisplays
      // so we can initialize our invisibleCount and add listeners.
      const parentTrail = this.parent!.trail!;
      for ( let i = parentTrail.length; i < trail.length; i++ ) {
        const relativeNode = trail.nodes[ i ];
        this.relativeNodes.push( relativeNode );

        const pdomDisplays = relativeNode._pdomDisplaysInfo.pdomDisplays;
        const isVisible = _.includes( pdomDisplays, display );
        this.relativeVisibilities.push( isVisible );
        if ( !isVisible ) {
          this.invisibleCount++;
        }

        const listener = this.checkAccessibleDisplayVisibility.bind( this, i - parentTrail.length );
        relativeNode.pdomDisplaysEmitter.addListener( listener );
        this.relativeListeners.push( listener );
      }

      this.updateVisibility();
    }

    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.PDOMInstance(
      `Initialized ${this.toString()}` );

    return this;
  }

  /**
   * Adds a series of (sorted) accessible instances as children.
   */
  public addConsecutiveInstances( pdomInstances: PDOMInstance[] ): void {
    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.PDOMInstance(
      `addConsecutiveInstances on ${this.toString()} with: ${pdomInstances.map( inst => inst.toString() ).join( ',' )}` );
    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.push();

    const hadChildren = this.children.length > 0;

    Array.prototype.push.apply( this.children, pdomInstances );

    for ( let i = 0; i < pdomInstances.length; i++ ) {
      // Append the container parent to the end (so that, when provided in order, we don't have to resort below
      // when initializing).
      const placeableSibling = this.peer!.getPlaceableSibling();
      assert && assert( !!placeableSibling, 'Primary sibling must be defined to insert elements.' );

      // @ts-expect-error - when PDOMPeer is converted to TS this ts-expect-error can probably be removed
      PDOMUtils.insertElements( placeableSibling, pdomInstances[ i ].peer.topLevelElements );
    }

    if ( hadChildren ) {
      this.sortChildren();
    }

    if ( assert && this.node ) {
      assert && assert( !!this.node );

      // We do not support rendering children into a Node that has innerContent.
      // If you hit this when mutating both children and innerContent at the same time, it is an issue with scenery.
      // Remove one in a single step and them add then other in the next step.
      this.children.length > 0 && assert( !this.node.innerContent,
        `${this.children.length} child PDOMInstances present but this node has innerContent: ${this.node.innerContent}` );
    }

    if ( UNIQUE_ID_STRATEGY === PDOMUniqueIdStrategy.INDICES ) {

      // This kills performance if there are enough PDOMInstances
      this.updateDescendantPeerIds( pdomInstances );
    }

    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.pop();
  }

  /**
   * Removes any child instances that are based on the provided trail.
   */
  public removeInstancesForTrail( trail: Trail ): void {
    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.PDOMInstance(
      `removeInstancesForTrail on ${this.toString()} with trail ${trail.toString()}` );
    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.push();

    for ( let i = 0; i < this.children.length; i++ ) {
      const childInstance = this.children[ i ];
      const childTrail = childInstance.trail;

      // Not worth it to inspect before our trail ends, since it should be (!) guaranteed to be equal
      let differs = childTrail!.length < trail.length;
      if ( !differs ) {
        for ( let j = this.trail!.length; j < trail.length; j++ ) {
          if ( trail.nodes[ j ] !== childTrail!.nodes[ j ] ) {
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

    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.pop();
  }

  /**
   * Removes all of the children.
   */
  public removeAllChildren(): void {
    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.PDOMInstance( `removeAllChildren on ${this.toString()}` );
    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.push();

    while ( this.children.length ) {
      this.children.pop()!.dispose();
    }

    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.pop();
  }

  /**
   * Returns a PDOMInstance child (if one exists with the given Trail), or null otherwise.
   */
  public findChildWithTrail( trail: Trail ): PDOMInstance | null {
    for ( let i = 0; i < this.children.length; i++ ) {
      const child = this.children[ i ];
      if ( child.trail!.equals( trail ) ) {
        return child;
      }
    }
    return null;
  }

  /**
   * Remove a subtree of PDOMInstances from this PDOMInstance
   *
   * @param trail - children of this PDOMInstance will be removed if the child trails are extensions
   *                        of the trail.
   * (scenery-internal)
   */
  public removeSubtree( trail: Trail ): void {
    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.PDOMInstance(
      `removeSubtree on ${this.toString()} with trail ${trail.toString()}` );
    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.push();

    for ( let i = this.children.length - 1; i >= 0; i-- ) {
      const childInstance = this.children[ i ];
      if ( childInstance.trail!.isExtensionOf( trail, true ) ) {
        sceneryLog && sceneryLog.PDOMInstance && sceneryLog.PDOMInstance(
          `Remove parent: ${this.toString()}, child: ${childInstance.toString()}` );
        this.children.splice( i, 1 ); // remove it from the children array

        // Dispose the entire subtree of PDOMInstances
        childInstance.dispose();
      }
    }

    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.pop();
  }

  /**
   * Checks to see whether our visibility needs an update based on a pdomDisplays change.
   *
   * @param index - Index into the relativeNodes array (which node had the notification)
   */
  private checkAccessibleDisplayVisibility( index: number ): void {
    const isNodeVisible = _.includes( this.relativeNodes![ index ]._pdomDisplaysInfo.pdomDisplays, this.display );
    const wasNodeVisible = this.relativeVisibilities[ index ];

    if ( isNodeVisible !== wasNodeVisible ) {
      this.relativeVisibilities[ index ] = isNodeVisible;

      const wasVisible = this.invisibleCount === 0;

      this.invisibleCount += ( isNodeVisible ? -1 : 1 );
      assert && assert( this.invisibleCount >= 0 && this.invisibleCount <= this.relativeNodes!.length );

      const isVisible = this.invisibleCount === 0;

      if ( isVisible !== wasVisible ) {
        this.updateVisibility();
      }
    }
  }

  /**
   * Update visibility of this peer's accessible DOM content. The hidden attribute will hide all of the descendant
   * DOM content, so it is not necessary to update the subtree of PDOMInstances since the browser
   * will do this for us.
   */
  private updateVisibility(): void {
    assert && assert( !!this.peer, 'Peer needs to be available on update visibility.' );
    this.peer!.setVisible( this.invisibleCount <= 0 );

    // if we hid a parent element, blur focus if active element was an ancestor
    if ( !this.peer!.isVisible() && getPDOMFocusedNode()?.pdomInstances[ 0 ]?.trail!.containsNode( this.node! ) ) {
      FocusManager.pdomFocus = null;
    }
  }

  /**
   * Returns whether the parallel DOM for this instance and its ancestors are not hidden.
   */
  public isGloballyVisible(): boolean {
    assert && assert( !!this.peer, 'PDOMPeer needs to be available, has this PDOMInstance been disposed?' );

    // If this peer is hidden, then return because that attribute will bubble down to children,
    // otherwise recurse to parent.
    if ( !this.peer!.isVisible() ) {
      return false;
    }
    else if ( this.parent ) {
      return this.parent.isGloballyVisible();
    }
    else { // base case at root
      return true;
    }
  }

  /**
   * Returns what our list of children (after sorting) should be.
   *
   * @param trail - A partial trail, where the root of the trail is either this.node or the display's root
   *                        node (if we are the root PDOMInstance)
   */
  private getChildOrdering( trail: Trail ): PDOMInstance[] {
    const node = trail.lastNode();
    const effectiveChildren = node.getEffectiveChildren();
    let i;
    const instances: PDOMInstance[] = [];

    // base case, node has accessible content, but don't match the "root" node of this accessible instance
    if ( node.hasPDOMContent && node !== this.node ) {
      const potentialInstances = node.pdomInstances;

      instanceLoop: // eslint-disable-line no-labels
        for ( i = 0; i < potentialInstances.length; i++ ) {
          const potentialInstance = potentialInstances[ i ];
          if ( potentialInstance.parent !== this ) {
            continue;
          }

          for ( let j = 0; j < trail.length; j++ ) {
            if ( trail.nodes[ j ] !== potentialInstance.trail!.nodes[ j + potentialInstance.trail!.length - trail.length ] ) {
              continue instanceLoop; // eslint-disable-line no-labels
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
  }

  /**
   * Sort our child accessible instances in the order they should appear in the parallel DOM. We do this by
   * creating a comparison function between two accessible instances. The function walks along the trails
   * of the children, looking for specified accessible orders that would determine the ordering for the two
   * PDOMInstances.
   *
   * (scenery-internal)
   */
  public sortChildren(): void {
    // It's simpler/faster to just grab our order directly with one recursion, rather than specifying a sorting
    // function (since a lot gets re-evaluated in that case).

    assert && assert( this.peer !== null, 'peer required for sort' );
    let nodeForTrail: Node;
    if ( this.isRootInstance ) {

      assert && assert( this.display !== null, 'Display should be available for the root' );
      nodeForTrail = this.display!.rootNode;
    }
    else {
      assert && assert( this.node !== null, 'Node should be defined, were we disposed?' );
      nodeForTrail = this.node!;
    }
    const targetChildren = this.getChildOrdering( new Trail( nodeForTrail ) );

    assert && assert( targetChildren.length === this.children.length, 'sorting should not change number of children' );

    // {Array.<PDOMInstance>}
    this.children = targetChildren;

    // the DOMElement to add the child DOMElements to.
    const placeableSibling = this.peer!.getPlaceableSibling();

    // Ignore DAG for focused trail. We need to know if there is a focused child instance so that we can avoid
    // temporarily detaching the focused element from the DOM. See https://github.com/phetsims/my-solar-system/issues/142
    const focusedTrail = getPDOMFocusedNode()?.pdomInstances[ 0 ]?.trail || null;

    // "i" will keep track of the "collapsed" index when all DOMElements for all PDOMInstance children are
    // added to a single parent DOMElement (this PDOMInstance's PDOMPeer's primarySibling)
    let i = placeableSibling.childNodes.length - 1;

    const focusedChildInstance = focusedTrail && _.find( this.children, child => focusedTrail.containsNode( child.peer!.node ) );
    if ( focusedChildInstance ) {
      // If there's a focused child instance, we need to make sure that its primarySibling is not detached from the DOM
      // (this has caused focus issues, see https://github.com/phetsims/my-solar-system/issues/142).
      // Since this doesn't happen often, we can just recompute the full order, and move every other element.

      const desiredOrder = _.flatten( this.children.map( child => child.peer!.topLevelElements! ) );
      const needsOrderChange = !_.every( desiredOrder, ( desiredElement, index ) => placeableSibling.children[ index ] === desiredElement );

      if ( needsOrderChange ) {
        const pivotElement = focusedChildInstance.peer!.getTopLevelElementContainingPrimarySibling();
        const pivotIndex = desiredOrder.indexOf( pivotElement );
        assert && assert( pivotIndex >= 0 );

        // Insert all elements before the pivot element
        for ( let j = 0; j < pivotIndex; j++ ) {
          placeableSibling.insertBefore( desiredOrder[ j ], pivotElement );
        }

        // Insert all elements after the pivot element
        for ( let j = pivotIndex + 1; j < desiredOrder.length; j++ ) {
          placeableSibling.appendChild( desiredOrder[ j ] );
        }
      }
    }
    else {
      // Iterate through all PDOMInstance children
      for ( let peerIndex = this.children.length - 1; peerIndex >= 0; peerIndex-- ) {
        const peer = this.children[ peerIndex ].peer!;

        // Iterate through all top level elements of a PDOMInstance's peer
        for ( let elementIndex = peer.topLevelElements!.length - 1; elementIndex >= 0; elementIndex-- ) {
          const element = peer.topLevelElements![ elementIndex ];

          // Reorder DOM elements in a way that doesn't do any work if they are already in a sorted order.
          // No need to reinsert if `element` is already in the right order
          if ( placeableSibling.childNodes[ i ] !== element ) {
            placeableSibling.insertBefore( element, placeableSibling.childNodes[ i + 1 ] );
          }

          // Decrement so that it is easier to place elements using the browser's Node.insertBefore API
          i--;
        }
      }
    }

    if ( assert ) {
      const desiredOrder = _.flatten( this.children.map( child => child.peer!.topLevelElements! ) );

      // Verify the order
      assert( _.every( desiredOrder, ( desiredElement, index ) => placeableSibling.children[ index ] === desiredElement ) );
    }

    if ( UNIQUE_ID_STRATEGY === PDOMUniqueIdStrategy.INDICES ) {

      // This kills performance if there are enough PDOMInstances
      this.updateDescendantPeerIds( this.children );
    }
  }

  /**
   * Create a new TransformTracker that will observe transforms along the trail of this PDOMInstance OR
   * the provided pdomTransformSourceNode. See ParallelDOM.setPDOMTransformSourceNode(). The The source Node
   * must not use DAG so that its trail is unique.
   */
  public updateTransformTracker( pdomTransformSourceNode: Node | null ): void {
    this.transformTracker && this.transformTracker.dispose();

    let trackedTrail = null;
    if ( pdomTransformSourceNode ) {
      trackedTrail = pdomTransformSourceNode.getUniqueTrail();
    }
    else {
      trackedTrail = guessVisualTrail( this.trail!, this.display!.rootNode );
    }

    this.transformTracker = new TransformTracker( trackedTrail );
  }

  /**
   * Depending on what the unique ID strategy is, formulate the correct id for this PDOM instance.
   */
  public getPDOMInstanceUniqueId(): string {

    if ( UNIQUE_ID_STRATEGY === PDOMUniqueIdStrategy.INDICES ) {

      const indicesString = [];

      let pdomInstance: PDOMInstance = this; // eslint-disable-line consistent-this, @typescript-eslint/no-this-alias

      while ( pdomInstance.parent ) {
        const indexOf = pdomInstance.parent.children.indexOf( pdomInstance );
        if ( indexOf === -1 ) {
          return 'STILL_BEING_CREATED' + dotRandom.nextDouble();
        }
        indicesString.unshift( indexOf );
        pdomInstance = pdomInstance.parent;
      }
      return indicesString.join( PDOM_UNIQUE_ID_SEPARATOR );
    }
    else {
      assert && assert( UNIQUE_ID_STRATEGY === PDOMUniqueIdStrategy.TRAIL_ID );

      return this.trail!.getUniqueId();
    }
  }

  /**
   * Using indices requires updating whenever the PDOMInstance tree changes, so recursively update all descendant
   * ids from such a change. Update peer ids for provided instances and all descendants of provided instances.
   */
  private updateDescendantPeerIds( pdomInstances: PDOMInstance[] ): void {
    assert && assert( UNIQUE_ID_STRATEGY === PDOMUniqueIdStrategy.INDICES, 'method should not be used with uniqueId comes from TRAIL_ID' );
    const toUpdate = Array.from( pdomInstances );
    while ( toUpdate.length > 0 ) {
      const pdomInstance = toUpdate.shift()!;
      pdomInstance.peer!.updateIndicesStringAndElementIds();
      toUpdate.push( ...pdomInstance.children );
    }
  }

  /**
   * Recursive disposal, to make eligible for garbage collection.
   *
   * (scenery-internal)
   */
  public dispose(): void {
    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.PDOMInstance(
      `Disposing ${this.toString()}` );
    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.push();

    assert && assert( !!this.peer, 'PDOMPeer required, were we already disposed?' );
    const thisPeer = this.peer!;

    // Disconnect DOM and remove listeners
    if ( !this.isRootInstance ) {

      // remove this peer's primary sibling DOM Element (or its container parent) from the parent peer's
      // primary sibling (or its child container)
      PDOMUtils.removeElements( this.parent!.peer!.getPlaceableSibling(), thisPeer.topLevelElements! );

      for ( let i = 0; i < this.relativeNodes!.length; i++ ) {
        this.relativeNodes![ i ].pdomDisplaysEmitter.removeListener( this.relativeListeners[ i ] );
      }
    }

    while ( this.children.length ) {
      this.children.pop()!.dispose();
    }

    // NOTE: We dispose OUR peer after disposing children, so our peer can be available for our children during
    // disposal.
    thisPeer.dispose();

    // dispose after the peer so the peer can remove any listeners from it
    this.transformTracker!.dispose();
    this.transformTracker = null;

    // If we are the root accessible instance, we won't actually have a reference to a node.
    if ( this.node ) {
      this.node.removePDOMInstance( this );
    }

    this.relativeNodes = null;
    this.display = null;
    this.trail = null;
    this.node = null;

    this.peer = null;
    this.isDisposed = true;

    this.freeToPool();

    sceneryLog && sceneryLog.PDOMInstance && sceneryLog.pop();
  }

  /**
   * For debugging purposes.
   */
  public toString(): string {
    return `${this.id}#{${this.trail!.toString()}}`;
  }

  /**
   * For debugging purposes, inspect the tree of PDOMInstances from the root.
   *
   * Only ever called from the _rootPDOMInstance of the display.
   *
   * (scenery-internal)
   */
  public auditRoot(): void {
    if ( assert ) {
      const rootNode = this.display!.rootNode;

      assert( this.trail!.length === 0,
        'Should only call auditRoot() on the root PDOMInstance for a display' );

      function audit( fakeInstance: FakeInstance, pdomInstance: PDOMInstance ): void {
        assert && assert( fakeInstance.children.length === pdomInstance.children.length,
          'Different number of children in accessible instance' );

        assert && assert( fakeInstance.node === pdomInstance.node, 'Node mismatch for PDOMInstance' );

        for ( let i = 0; i < pdomInstance.children.length; i++ ) {
          audit( fakeInstance.children[ i ], pdomInstance.children[ i ] );
        }

        const isVisible = pdomInstance.isGloballyVisible();

        let shouldBeVisible = true;
        for ( let i = 0; i < pdomInstance.trail!.length; i++ ) {
          const node = pdomInstance.trail!.nodes[ i ];
          const trails = node.getTrailsTo( rootNode ).filter( trail => trail.isPDOMVisible() );
          if ( trails.length === 0 ) {
            shouldBeVisible = false;
            break;
          }
        }

        assert && assert( isVisible === shouldBeVisible, 'Instance visibility mismatch' );
      }

      audit( PDOMInstance.createFakePDOMTree( rootNode ), this );
    }
  }

  /**
   * Creates a fake PDOMInstance-like tree structure (with the equivalent nodes and children structure).
   * For debugging.
   *
   * @returns Type FakePDOMInstance: { node: {Node}, children: {Array.<FakePDOMInstance>} }
   */
  private static createFakePDOMTree( rootNode: Node ): FakeInstance {
    function createFakeTree( node: Node ): object {
      let fakeInstances = _.flatten( node.getEffectiveChildren().map( createFakeTree ) ) as FakeInstance[];
      if ( node.hasPDOMContent ) {
        fakeInstances = [ {
          node: node,
          children: fakeInstances
        } ];
      }
      return fakeInstances;
    }

    return {
      node: null,

      // @ts-expect-error
      children: createFakeTree( rootNode )
    };
  }

  public freeToPool(): void {
    PDOMInstance.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( PDOMInstance, {
    initialize: PDOMInstance.prototype.initializePDOMInstance
  } );
}

scenery.register( 'PDOMInstance', PDOMInstance );


export default PDOMInstance;