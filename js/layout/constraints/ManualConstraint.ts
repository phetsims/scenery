// Copyright 2021-2022, University of Colorado Boulder

/**
 * ManualConstraint exists for cases where imperative-based positioning code (e.g. `node.left = otherNode.right + 5`)
 * is best for a case, and should be rerun whenever one of the nodes changes bounds.
 *
 * ManualConstraint also can handle cases where the nodes do not live in the same coordinate frame (but instead with
 * some common ancestor).
 *
 * For example:
 *
 * new ManualConstraint( [ firstNode, secondNode ], ( firstProxy, secondProxy ) => {
 *   firstProxy.left = secondProxy.right + 5;
 *   secondProxy.centerY = firstProxy.centerY;
 * } );
 *
 * Notably in the callback, it uses LayoutProxy (which has the positional getters/setters of an object, and handles
 * coordinate transforms).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Node, LayoutConstraint, LayoutProxy, LayoutCell } from '../../imports.js';

// Turns a tuple of things into a tuple of LayoutProxies
type LayoutProxyMap<T> = {
  [Property in keyof T]: LayoutProxy // eslint-disable-line
};
type LayoutCallback<T extends any[]> = ( ...args: LayoutProxyMap<T> ) => void;

export default class ManualConstraint<T extends Node[]> extends LayoutConstraint {

  private readonly nodes: T;
  private readonly cells: LayoutCell[];
  private readonly layoutCallback: LayoutCallback<T>;

  constructor( ancestorNode: Node, nodes: T, layoutCallback: LayoutCallback<T> ) {

    assert && assert( ancestorNode instanceof Node );
    assert && assert( Array.isArray( nodes ) && _.every( nodes, node => node instanceof Node ) );
    assert && assert( typeof layoutCallback === 'function' );

    super( ancestorNode );

    this.lock();

    this.nodes = nodes;

    // Having cells will give us proxy Properties and listening for when it's added for free
    this.cells = nodes.map( node => new LayoutCell( this, node, null ) );

    this.layoutCallback = layoutCallback;

    // Hook up to listen to these nodes
    this.nodes.forEach( node => this.addNode( node, false ) );

    // Run the layout manually at the start
    this.unlock();
    this.updateLayout();
  }

  override layout(): void {
    super.layout();

    assert && assert( _.every( this.nodes, node => !node.isDisposed ) );

    const isMissingProxy = _.some( this.cells, cell => !cell.isConnected() );
    if ( !isMissingProxy ) {
      const proxies = this.cells.map( cell => cell.proxy );

      this.layoutCallback.apply( null, proxies as LayoutProxyMap<T> );

      this.finishedLayoutEmitter.emit();
    }
  }

  /**
   * Releases references
   */
  override dispose(): void {
    this.cells.forEach( cell => cell.dispose() );

    super.dispose();
  }

  static create<T extends Node[]>( ancestorNode: Node, nodes: T, layoutCallback: LayoutCallback<T> ): ManualConstraint<T> {
    return new ManualConstraint( ancestorNode, nodes, layoutCallback );
  }
}

scenery.register( 'ManualConstraint', ManualConstraint );
