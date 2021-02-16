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

    // @private {boolean} - Prevents layout() from running while true. Generally will be unlocked and laid out.
    this._updateLayoutLocked = false;

    // @private {function}
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
   * Updates the layout of this LayoutBox. Called automatically during initialization, when children change (if
   * resize is true), or when client wants to call this public method for any reason.
   * @public
   */
  updateLayout() {
    if ( !this._updateLayoutLocked ) {
      this._updateLayoutLocked = true;
      this.layout();
      this._updateLayoutLocked = false;
    }
  }

  /**
   * Called when we attempt to automatically layout components.
   * @private
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