// Copyright 2021-2022, University of Colorado Boulder

/**
 * Abstract supertype for layout constraints.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import { scenery, LayoutProxy, Node, isWidthSizable, isHeightSizable } from '../imports.js';

class LayoutConstraint {

  // The Node in whose local coordinate frame our layout computations are done.
  private ancestorNode: Node;

  // Prevents layout() from running while true. Generally will be unlocked and laid out.
  private _layoutLockCount: number;

  // Whether there was a layout attempt during the lock
  private _layoutAttemptDuringLock: boolean;

  private _enabled: boolean;

  protected _updateLayoutListener: () => void;

  private _listenedNodes: Set<Node>;

  // scenery-internal
  finishedLayoutEmitter: TinyEmitter<[]>;

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

  addNode( node: Node ) {
    assert && assert( node instanceof Node );
    assert && assert( !this._listenedNodes.has( node ) );

    // TODO: listen or un-listen based on whether we are enabled?
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

  removeNode( node: Node ) {
    assert && assert( node instanceof Node );
    assert && assert( this._listenedNodes.has( node ) );

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

  lock() {
    this._layoutLockCount++;
  }

  unlock() {
    this._layoutLockCount--;
  }

  /**
   * Updates the layout of this constraint. Called automatically during initialization, when children change (if
   * resize is true), or when client wants to call this public method for any reason.
   */
  updateLayout() {
    let count = 0;

    if ( this.isLocked ) {
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
  updateLayoutAutomatically() {
    if ( this._enabled ) {
      this.updateLayout();
    }
  }

  createLayoutProxy( node: Node ): LayoutProxy {
    assert && assert( node instanceof Node );

    // TODO: How to handle the case where there is no trail?

    return LayoutProxy.createFromPool( node.getUniqueTrailTo( this.ancestorNode ).removeAncestor() );
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
  dispose() {
    // Clean up listeners to any listened nodes
    const listenedNodes = [ ...this._listenedNodes.keys() ];
    for ( let i = 0; i < listenedNodes.length; i++ ) {
      this.removeNode( listenedNodes[ i ] );
    }
  }
}

scenery.register( 'LayoutConstraint', LayoutConstraint );
export default LayoutConstraint;