// Copyright 2021-2023, University of Colorado Boulder

/**
 * Abstract supertype for layout constraints. Provides a lot of assistance to layout handling, including adding/removing
 * listeners, and reentrancy detection/loop prevention.
 *
 * We'll also handle reentrancy somewhat specially. If code tries to enter a layout reentrantly (while a layout is
 * already executing), we'll instead IGNORE this second one (and set a flag). Once our first layout is done, we'll
 * attempt to run the layout again. In case the subtype needs to lock multiple times (if a layout is FORCED), we have
 * an integer count of how many "layout" calls we're in (_layoutLockCount). Once this reaches zero, we're effectively
 * unlocked and not inside any layout calls.
 *
 * NOTE: This can still trigger infinite loops nominally (if every layout call triggers another layout call), but we
 * have a practical assertion limit that will stop this and flag it as an error.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TEmitter from '../../../../axon/js/TEmitter.js';
import TinyEmitter from '../../../../axon/js/TinyEmitter.js';
import { HeightSizableNode, LayoutProxy, extendsHeightSizable, extendsWidthSizable, Node, scenery, SizableNode, WidthSizableNode } from '../../imports.js';

export default abstract class LayoutConstraint {

  // The Node in whose local coordinate frame our layout computations are done.
  public readonly ancestorNode: Node;

  // Prevents layout() from running while greater than zero. Generally will be unlocked and laid out.
  // See the documentation at the top of the file for more on reentrancy.
  private _layoutLockCount = 0;

  // Whether there was a layout attempt during the lock
  private _layoutAttemptDuringLock = false;

  // When we are disabled (say, a layout container has resize:false), we won't automatically do layout
  private _enabled = true;

  protected readonly _updateLayoutListener: () => void;

  // Track Nodes we're listening to (for memory cleanup purposes)
  private readonly _listenedNodes: Set<Node> = new Set<Node>();

  // (scenery-internal) - emits when we've finished layout
  public readonly finishedLayoutEmitter: TEmitter = new TinyEmitter();

  /**
   * (scenery-internal)
   */
  protected constructor( ancestorNode: Node ) {
    this.ancestorNode = ancestorNode;
    this._updateLayoutListener = this.updateLayoutAutomatically.bind( this );
  }

  /**
   * Adds listeners to a Node, so that our layout updates will happen if this Node's Properties
   * (bounds/visibility/minimum size) change. Will be cleared on disposal of this type.
   * (scenery-internal)
   *
   * @param node
   * @param addLock - If true, we'll mark the node as having this layout constraint as responsible for its layout.
   * It will be an assertion failure if another layout container tries to lock the same node (so that we don't run into
   * infinite loops where multiple constraints try to move a node back-and-forth).
   * See Node's _activeParentLayoutConstraint for more information.
   */
  public addNode( node: Node, addLock = true ): void {
    assert && assert( !this._listenedNodes.has( node ) );
    assert && assert( !addLock || !node._activeParentLayoutConstraint, 'This node is already managed by a layout container - make sure to wrap it in a Node if DAG, removing it from an old layout container, etc.' );

    if ( addLock ) {
      node._activeParentLayoutConstraint = this;
    }

    node.boundsProperty.lazyLink( this._updateLayoutListener );
    node.visibleProperty.lazyLink( this._updateLayoutListener );
    if ( extendsWidthSizable( node ) ) {
      node.minimumWidthProperty.lazyLink( this._updateLayoutListener );
      node.isWidthResizableProperty.lazyLink( this._updateLayoutListener );
    }
    if ( extendsHeightSizable( node ) ) {
      node.minimumHeightProperty.lazyLink( this._updateLayoutListener );
      node.isHeightResizableProperty.lazyLink( this._updateLayoutListener );
    }

    this._listenedNodes.add( node );
  }

  /**
   * (scenery-internal)
   */
  public removeNode( node: Node ): void {
    assert && assert( this._listenedNodes.has( node ) );

    // Optional, since we might not have added the "lock" in addNode
    if ( node._activeParentLayoutConstraint === this ) {
      node._activeParentLayoutConstraint = null;
    }

    node.boundsProperty.unlink( this._updateLayoutListener );
    node.visibleProperty.unlink( this._updateLayoutListener );
    if ( extendsWidthSizable( node ) ) {
      node.minimumWidthProperty.unlink( this._updateLayoutListener );
      node.isWidthResizableProperty.unlink( this._updateLayoutListener );
    }
    if ( extendsHeightSizable( node ) ) {
      node.minimumHeightProperty.unlink( this._updateLayoutListener );
      node.isHeightResizableProperty.unlink( this._updateLayoutListener );
    }

    this._listenedNodes.delete( node );
  }

  /**
   * NOTE: DO NOT call from places other than super.layout() in overridden layout() OR from the existing call in
   *       updateLayout(). Doing so would break the lock mechanism.
   * NOTE: Cannot be marked as abstract due to how mixins work
   */
  protected layout(): void {
    // See subclass for implementation
  }

  /**
   * (scenery-internal)
   */
  public get isLocked(): boolean {
    return this._layoutLockCount > 0;
  }

  /**
   * Locks the layout, so that automatic layout will NOT be triggered synchronously until unlock() is called and
   * the lock count returns to 0. This is set up so that if we trigger multiple reentrancy, we will only attempt to
   * re-layout once ALL of the layouts are finished.
   * (scenery-internal)
   */
  public lock(): void {
    this._layoutLockCount++;
  }

  /**
   * Unlocks the layout. Generally (but not always), updateLayout() or updateLayoutAutomatically() should be called
   * after this, as locks are generally used for this purpose.
   * (scenery-internal)
   */
  public unlock(): void {
    this._layoutLockCount--;
  }

  /**
   * Here for manual validation (say, in the devtools) - While some layouts are going on, this may not be correct, so it
   * could not be added to post-layout validation.
   * (scenery-internal)
   */
  public validateLocalPreferredWidth( layoutContainer: WidthSizableNode ): void {
    if ( assert && layoutContainer.localBounds.isFinite() && !this._layoutAttemptDuringLock ) {
      layoutContainer.validateLocalPreferredWidth();
    }
  }

  /**
   * Here for manual validation (say, in the devtools) - While some layouts are going on, this may not be correct, so it
   * could not be added to post-layout validation.
   * (scenery-internal)
   */
  public validateLocalPreferredHeight( layoutContainer: HeightSizableNode ): void {
    if ( assert && layoutContainer.localBounds.isFinite() && !this._layoutAttemptDuringLock ) {
      layoutContainer.validateLocalPreferredHeight();
    }
  }

  /**
   * Here for manual validation (say, in the devtools) - While some layouts are going on, this may not be correct, so it
   * could not be added to post-layout validation.
   * (scenery-internal)
   */
  public validateLocalPreferredSize( layoutContainer: SizableNode ): void {
    if ( assert && layoutContainer.localBounds.isFinite() && !this._layoutAttemptDuringLock ) {
      layoutContainer.validateLocalPreferredSize();
    }
  }

  /**
   * Updates the layout of this constraint. Called automatically during initialization, when children change (if
   * resize is true), or when client wants to call this public method for any reason.
   */
  public updateLayout(): void {
    let count = 0;

    // If we're locked AND someone tries to do layout, record this so we can attempt layout once we are not locked
    // anymore. We have some infinite-loop detection here for common development errors.
    if ( this.isLocked ) {
      assert && count++;
      assert && assert( ++count < 500, 'Likely infinite loop detected, are we triggering layout within the layout?' );
      this._layoutAttemptDuringLock = true;
    }
    else {
      this.lock();

      // Run layout until we didn't get a layout attempt during our last attempt. This component's layout should now
      // be correct and stable.
      do {
        this._layoutAttemptDuringLock = false;
        this.layout();
      }
        // If we got any layout attempts during the lock, we'll want to rerun the layout
      while ( this._layoutAttemptDuringLock );

      this.unlock();
    }
  }

  /**
   * Called when we attempt to automatically layout components. (scenery-internal)
   */
  public updateLayoutAutomatically(): void {
    if ( this._enabled ) {
      this.updateLayout();
    }
  }

  /**
   * Creates a LayoutProxy for a unique trail from our ancestorNode to this Node (or null if that's not possible)
   * (scenery-internal)
   */
  public createLayoutProxy( node: Node ): LayoutProxy | null {
    const trails = node.getTrails( n => n === this.ancestorNode );

    if ( trails.length === 1 ) {
      return LayoutProxy.pool.create( trails[ 0 ].removeAncestor() );
    }
    else {
      return null;
    }
  }

  public get enabled(): boolean {
    return this._enabled;
  }

  public set enabled( value: boolean ) {
    if ( this._enabled !== value ) {
      this._enabled = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * Releases references
   */
  public dispose(): void {
    // Clean up listeners to any listened nodes
    const listenedNodes = [ ...this._listenedNodes.keys() ];
    for ( let i = 0; i < listenedNodes.length; i++ ) {
      this.removeNode( listenedNodes[ i ] );
    }

    this.finishedLayoutEmitter.dispose();
  }
}

scenery.register( 'LayoutConstraint', LayoutConstraint );
