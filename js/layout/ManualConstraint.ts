// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Node, LayoutConstraint, LayoutProxy } from '../imports.js';

// Turns a tuple of things into a tuple of LayoutProxies
type LayoutProxyMap<T> = {
  [Property in keyof T]: LayoutProxy // eslint-disable-line
};
type LayoutCallback<T extends any[]> = ( ...args: LayoutProxyMap<T> ) => void;

class ManualConstraint<T extends Node[]> extends LayoutConstraint {

  private nodes: T;
  private layoutCallback: LayoutCallback<T>;

  // Minimizing garbage created
  private proxyFactory: ( n: Node ) => LayoutProxy;

  constructor( ancestorNode: Node, nodes: T, layoutCallback: LayoutCallback<T> ) {

    assert && assert( ancestorNode instanceof Node );
    assert && assert( Array.isArray( nodes ) && _.every( nodes, node => node instanceof Node ) );
    assert && assert( typeof layoutCallback === 'function' );

    super( ancestorNode );

    this.nodes = nodes;
    this.layoutCallback = layoutCallback;
    this.proxyFactory = this.createLayoutProxy.bind( this );

    // Hook up to listen to these nodes
    this.nodes.forEach( node => this.addNode( node ) );

    // Run the layout manually at the start
    this.updateLayout();
  }

  layout() {
    super.layout();

    assert && assert( _.every( this.nodes, node => !node.isDisposed ) );

    const proxies = this.nodes.map( this.proxyFactory );

    this.layoutCallback.apply( null, proxies as LayoutProxyMap<T> );

    // Minimizing garbage created
    for ( let i = 0; i < proxies.length; i++ ) {
      proxies[ i ].dispose();
    }

    this.finishedLayoutEmitter.emit();
  }

  static create<T extends Node[]>( ancestorNode: Node, nodes: T, layoutCallback: LayoutCallback<T> ): ManualConstraint<T> {
    return new ManualConstraint( ancestorNode, nodes, layoutCallback );
  }
}

scenery.register( 'ManualConstraint', ManualConstraint );

export default ManualConstraint;