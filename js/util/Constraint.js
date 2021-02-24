// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import LayoutProxy from './LayoutProxy.js';

class Constraint {
  /**
   * @param {Node} rootNode
   */
  constructor( rootNode ) {
    assert && assert( rootNode instanceof Node );

    // @private {Node}
    this.rootNode = rootNode;

    // @private {boolean}
    this._enabled = true;

    // @private {number} - Prevents layout() from running while true. Generally will be unlocked and laid out.
    this._layoutLockCount = 0;

    // @protected {function}
    this._updateLayoutListener = this.updateLayoutAutomatically.bind( this );

    // @private {Set.<Node>}
    this._listenedNodes = new Set();
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
   * Updates the layout of this LayoutBox. Called automatically during initialization, when children change (if
   * resize is true), or when client wants to call this public method for any reason.
   * @public
   */
  updateLayout() {
    if ( !this.isLocked ) {
      this.lock();

      this.layout();

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
    // TODO: How to handle the case where there is no trail?

    return LayoutProxy.createFromPool( node.getUniqueTrailTo( this.rootNode ).removeAncestor() );
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

scenery.register( 'Constraint', Constraint );
export default Constraint;