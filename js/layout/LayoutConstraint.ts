// Copyright 2021-2022, University of Colorado Boulder

/**
 * Abstract supertype for layout constraints. Provides a lot of assistance to layout handling, including adding/removing
 * listeners, and reentrancy detection/loop prevention.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import { scenery, LayoutProxy, Node, isWidthSizable, isHeightSizable } from '../imports.js';

export default class LayoutConstraint {

  // The Node in whose local coordinate frame our layout computations are done.
  readonly ancestorNode: Node;

  // Prevents layout() from running while true. Generally will be unlocked and laid out.
  private _layoutLockCount: number;

  // Whether there was a layout attempt during the lock
  private _layoutAttemptDuringLock: boolean;

  private _enabled: boolean;

  protected readonly _updateLayoutListener: () => void;

  private readonly _listenedNodes: Set<Node>;

  // scenery-internal
  readonly finishedLayoutEmitter: TinyEmitter<[]>;

  constructor( ancestorNode: Node ) {
    assert && assert( ancestorNode instanceof Node );

    this.ancestorNode = ancestorNode;
    this._enabled = true;
    this._layoutLockCount = 0;
    this._layoutAttemptDuringLock = false;
    this._updateLayoutListener = this.updateLayoutAutomatically.bind( this );
    this._listenedNodes = new Set();
    this.finishedLayoutEmitter = new TinyEmitter();
  }

  /**
   * Adds listeners to a Node, so that our layout updates will happen if this Node's properties
   * (bounds/visibility/minimum size) change. Will be cleared on disposal of this type.
   */
  addNode( node: Node, addLock = true ): void {
    assert && assert( node instanceof Node );
    assert && assert( !this._listenedNodes.has( node ) );
    assert && assert( !addLock || !node._activeParentLayoutConstraint, 'This node is already managed by a layout container - make sure to wrap it in a Node if DAG, removing it from an old layout container, etc.' );

    if ( addLock ) {
      node._activeParentLayoutConstraint = this;
    }
    // TODO: listen to things in-between!!
    node.boundsProperty.lazyLink( this._updateLayoutListener );
    node.visibleProperty.lazyLink( this._updateLayoutListener );
    if ( isWidthSizable( node ) ) {
      node.minimumWidthProperty.lazyLink( this._updateLayoutListener );
    }
    if ( isHeightSizable( node ) ) {
      node.minimumHeightProperty.lazyLink( this._updateLayoutListener );
    }

    this._listenedNodes.add( node );
  }

  removeNode( node: Node ): void {
    assert && assert( node instanceof Node );
    assert && assert( this._listenedNodes.has( node ) );

    // Optional, since we might not have added the "lock" in addNode
    if ( node._activeParentLayoutConstraint === this ) {
      node._activeParentLayoutConstraint = null;
    }

    node.boundsProperty.unlink( this._updateLayoutListener );
    node.visibleProperty.unlink( this._updateLayoutListener );
    if ( isWidthSizable( node ) ) {
      node.minimumWidthProperty.unlink( this._updateLayoutListener );
    }
    if ( isHeightSizable( node ) ) {
      node.minimumHeightProperty.unlink( this._updateLayoutListener );
    }

    this._listenedNodes.delete( node );
  }

  protected layout(): void {

  }

  get isLocked(): boolean {
    return this._layoutLockCount > 0;
  }

  lock(): void {
    this._layoutLockCount++;
  }

  unlock(): void {
    this._layoutLockCount--;
  }

  /**
   * Updates the layout of this constraint. Called automatically during initialization, when children change (if
   * resize is true), or when client wants to call this public method for any reason.
   */
  updateLayout(): void {
    let count = 0;

    if ( this.isLocked ) {
      assert && count++;
      assert && assert( ++count < 500, 'Likely infinite loop detected, are we triggering layout within the layout?' );
      this._layoutAttemptDuringLock = true;
    }
    else {
      this.lock();

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
  updateLayoutAutomatically(): void {
    if ( this._enabled ) {
      this.updateLayout();
    }
  }

  createLayoutProxy( node: Node ): LayoutProxy | null {
    assert && assert( node instanceof Node );

    // TODO: performance, don't create these closures
    const trails = node.getTrails( n => n === this.ancestorNode );

    if ( trails.length === 1 ) {
      return LayoutProxy.pool.create( trails[ 0 ].removeAncestor() );
    }
    else {
      return null;
    }
  }

  get enabled(): boolean {
    return this._enabled;
  }

  set enabled( value: boolean ) {
    assert && assert( typeof value === 'boolean' );

    if ( this._enabled !== value ) {
      this._enabled = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * Releases references
   */
  dispose(): void {
    // Clean up listeners to any listened nodes
    const listenedNodes = [ ...this._listenedNodes.keys() ];
    for ( let i = 0; i < listenedNodes.length; i++ ) {
      this.removeNode( listenedNodes[ i ] );
    }
  }
}

scenery.register( 'LayoutConstraint', LayoutConstraint );
