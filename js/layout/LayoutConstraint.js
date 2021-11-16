// Copyright 2021, University of Colorado Boulder

/**
 * Abstract supertype for layout constraints.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import LayoutProxy from './LayoutProxy.js';

class LayoutConstraint {
  /**
   * @param {Node} ancestorNode
   */
  constructor( ancestorNode ) {
    assert && assert( ancestorNode instanceof Node );

    // @private {Node}
    this.ancestorNode = ancestorNode;

    // @private {boolean}
    this._enabled = true;

    // @private {number} - Prevents layout() from running while true. Generally will be unlocked and laid out.
    this._layoutLockCount = 0;

    // @private {boolean} - Whether there was a layout attempt during the lock
    this._layoutAttemptDuringLock = false;

    // @protected {function}
    this._updateLayoutListener = this.updateLayoutAutomatically.bind( this );

    // @private {Set.<Node>}
    this._listenedNodes = new Set();

    // @public {TinyEmitter}
    this.finishedLayoutEmitter = new TinyEmitter();
  }

  /**
   * @public
   *
   * @param {Node} node
   */
  addNode( node ) {
    assert && assert( node instanceof Node );
    assert && assert( !this._listenedNodes.has( node ) );

    // TODO: listen or un-listen based on whether we are enabled?
    // TODO: listen to things in-between!!
    node.boundsProperty.lazyLink( this._updateLayoutListener );
    node.visibleProperty.lazyLink( this._updateLayoutListener );
    if ( node.widthSizable ) {
      node.minimumWidthProperty.lazyLink( this._updateLayoutListener );
    }
    if ( node.heightSizable ) {
      node.minimumHeightProperty.lazyLink( this._updateLayoutListener );
    }

    this._listenedNodes.add( node );
  }

  /**
   * @public
   *
   * @param {Node} node
   */
  removeNode( node ) {
    assert && assert( node instanceof Node );
    assert && assert( this._listenedNodes.has( node ) );

    node.boundsProperty.unlink( this._updateLayoutListener );
    node.visibleProperty.unlink( this._updateLayoutListener );
    if ( node.widthSizable ) {
      node.minimumWidthProperty.unlink( this._updateLayoutListener );
    }
    if ( node.heightSizable ) {
      node.minimumHeightProperty.unlink( this._updateLayoutListener );
    }

    this._listenedNodes.delete( node );
  }

  /**
   * @protected
   */
  layout() {

  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  get isLocked() {
    return this._layoutLockCount > 0;
  }

  /**
   * @public
   */
  lock() {
    this._layoutLockCount++;
  }

  /**
   * @public
   */
  unlock() {
    this._layoutLockCount--;
  }

  /**
   * Updates the layout of this constraint. Called automatically during initialization, when children change (if
   * resize is true), or when client wants to call this public method for any reason.
   * @public
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
   * Called when we attempt to automatically layout components.
   * @protected
   */
  updateLayoutAutomatically() {
    if ( this._enabled ) {
      this.updateLayout();
    }
  }

  /**
   * @public
   *
   * @param {Node} node
   * @returns {LayoutProxy}
   */
  createLayoutProxy( node ) {
    assert && assert( node instanceof Node );

    // TODO: How to handle the case where there is no trail?

    return LayoutProxy.createFromPool( node.getUniqueTrailTo( this.ancestorNode ).removeAncestor() );
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  get enabled() {
    return this._enabled;
  }

  /**
   * @public
   *
   * @param {boolean} value
   */
  set enabled( value ) {
    assert && assert( typeof value === 'boolean' );

    if ( this._enabled !== value ) {
      this._enabled = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    // Clean up listeners to any listened nodes
    const listenedNodes = this._listenedNodes.keys();
    for ( let i = 0; i < listenedNodes.length; i++ ) {
      this.removeNode( listenedNodes[ i ] );
    }
  }
}

scenery.register( 'LayoutConstraint', LayoutConstraint );
export default LayoutConstraint;