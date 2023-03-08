// Copyright 2023, University of Colorado Boulder

/**
 * A Property that will contain a set of all ancestor Nodes of a given Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyEmitter from '../../../axon/js/TinyEmitter.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import { Node, scenery } from '../imports.js';

export default class AncestorNodesProperty extends TinyProperty<Set<Node>> {

  // A set of nodes where we are listening to whether their parents change
  private readonly listenedNodeSet = new Set<Node>();

  private readonly _nodeUpdateListener: () => void;

  // Fired whenever we need to update the internal value (i.e. a parent was added or removed somewhere in the chain)
  public readonly updateEmitter = new TinyEmitter();

  public constructor( public readonly node: Node ) {
    super( new Set() );

    this._nodeUpdateListener = this.update.bind( this );

    // Listen to our own parent changes too (even though we aren't an ancestor)
    this.addNodeListener( node );

    this.update();
  }

  public override areValuesEqual( a: Set<Node>, b: Set<Node> ): boolean {
    // Don't fire notifications if it hasn't changed.
    return a.size === b.size && _.every( [ ...a ], node => b.has( node ) );
  }

  private update(): void {
    // Nodes that were touched in the scan (we should listen to changes to ANY of these to see if there is a connection
    // or disconnection). This could potentially cause our Property to change
    const nodeSet = new Set<Node>();

    // Recursively scan to identify all ancestors
    ( function recurse( node: Node ) {
      const parents = node.parents;

      parents.forEach( parent => {
        nodeSet.add( parent );
        recurse( parent );
      } );
    } )( this.node );

    // Add in new needed listeners
    nodeSet.forEach( node => {
      if ( !this.listenedNodeSet.has( node ) ) {
        this.addNodeListener( node );
      }
    } );

    // Remove listeners not needed anymore
    this.listenedNodeSet.forEach( node => {
      // NOTE: do NOT remove the listener that is listening to our node for changes (it's not an ancestor, and won't
      // come up in this list)
      if ( !nodeSet.has( node ) && node !== this.node ) {
        this.removeNodeListener( node );
      }
    } );

    this.value = nodeSet;

    this.updateEmitter.emit();
  }

  private addNodeListener( node: Node ): void {
    this.listenedNodeSet.add( node );
    node.parentAddedEmitter.addListener( this._nodeUpdateListener );
    node.parentRemovedEmitter.addListener( this._nodeUpdateListener );
  }

  private removeNodeListener( node: Node ): void {
    this.listenedNodeSet.delete( node );
    node.parentAddedEmitter.removeListener( this._nodeUpdateListener );
    node.parentRemovedEmitter.removeListener( this._nodeUpdateListener );
  }

  public override dispose(): void {
    this.listenedNodeSet.forEach( node => this.removeNodeListener( node ) );

    super.dispose();
  }
}

scenery.register( 'AncestorNodesProperty', AncestorNodesProperty );
