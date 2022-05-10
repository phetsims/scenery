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

import { scenery, Node, LayoutConstraint, LayoutProxy } from '../imports.js';

// Turns a tuple of things into a tuple of LayoutProxies
type LayoutProxyMap<T> = {
  [Property in keyof T]: LayoutProxy // eslint-disable-line
};
type LayoutCallback<T extends any[]> = ( ...args: LayoutProxyMap<T> ) => void;

export default class ManualConstraint<T extends Node[]> extends LayoutConstraint {

  private readonly nodes: T;
  private readonly layoutCallback: LayoutCallback<T>;

  // Minimizing garbage created
  private readonly proxyFactory: ( n: Node ) => LayoutProxy | null;

  constructor( ancestorNode: Node, nodes: T, layoutCallback: LayoutCallback<T> ) {

    assert && assert( ancestorNode instanceof Node );
    assert && assert( Array.isArray( nodes ) && _.every( nodes, node => node instanceof Node ) );
    assert && assert( typeof layoutCallback === 'function' );

    super( ancestorNode );

    this.nodes = nodes;
    this.layoutCallback = layoutCallback;
    this.proxyFactory = this.createLayoutProxy.bind( this );

    // Hook up to listen to these nodes
    this.nodes.forEach( node => this.addNode( node, false ) );

    // Run the layout manually at the start
    this.updateLayout();
  }

  override layout(): void {
    super.layout();

    assert && assert( _.every( this.nodes, node => !node.isDisposed ) );

    const proxies = this.nodes.map( this.proxyFactory );

    const hasNoNullProxy = _.every( proxies, proxy => proxy !== null );

    if ( hasNoNullProxy ) {
      this.layoutCallback.apply( null, proxies as LayoutProxyMap<T> );
    }

    // Minimizing garbage created
    for ( let i = 0; i < proxies.length; i++ ) {
      const proxy = proxies[ i ];
      proxy && proxy.dispose();
    }

    if ( hasNoNullProxy ) {
      this.finishedLayoutEmitter.emit();
    }
  }

  static create<T extends Node[]>( ancestorNode: Node, nodes: T, layoutCallback: LayoutCallback<T> ): ManualConstraint<T> {
    return new ManualConstraint( ancestorNode, nodes, layoutCallback );
  }
}

scenery.register( 'ManualConstraint', ManualConstraint );
