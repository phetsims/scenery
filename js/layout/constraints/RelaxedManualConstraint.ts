// Copyright 2021-2022, University of Colorado Boulder

/**
 * Like ManualConstraint, but permits layout when not all the nodes are connected (null will be passed through).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import IntentionalAny from '../../../../phet-core/js/types/IntentionalAny.js';
import { LayoutCell, LayoutConstraint, LayoutProxy, Node, scenery } from '../../imports.js';

// Turns a tuple of things into a tuple of LayoutProxies/null
type LayoutProxyMap<T> = {
  [Property in keyof T]: LayoutProxy | null
};
type LayoutCallback<T extends IntentionalAny[]> = ( ...args: LayoutProxyMap<T> ) => void;

export default class RelaxedManualConstraint<T extends Node[]> extends LayoutConstraint {

  private readonly nodes: T;

  // Cells provide us LayoutProxy and connection tracking
  private readonly cells: LayoutCell[];

  // The user-supplied callback that should be called to do layout
  private readonly layoutCallback: LayoutCallback<T>;

  public constructor( ancestorNode: Node, nodes: T, layoutCallback: LayoutCallback<T> ) {
    assert && assert( Array.isArray( nodes ) && _.every( nodes, node => node instanceof Node ) );

    super( ancestorNode );

    // Don't churn updates during construction
    this.lock();

    this.nodes = nodes;

    // Having cells will give us proxy Properties and listening for when it's added for free
    this.cells = nodes.map( node => new LayoutCell( this, node, null ) );

    this.layoutCallback = layoutCallback;

    // Hook up to listen to these nodes (will be handled by LayoutConstraint disposal)
    this.nodes.forEach( node => this.addNode( node, false ) );

    // Run the layout manually at the start
    this.unlock();
    this.updateLayout();
  }

  /**
   * (scenery-internal)
   */
  public override layout(): void {
    super.layout();

    assert && assert( _.every( this.nodes, node => !node.isDisposed ) );

    // If a cell is disconnected, pass in null
    const proxies = this.cells.map( cell => cell.isConnected() ? cell.proxy : null );

    this.layoutCallback.apply( null, proxies as LayoutProxyMap<T> );

    this.finishedLayoutEmitter.emit();
  }

  /**
   * Releases references
   */
  public override dispose(): void {
    this.cells.forEach( cell => cell.dispose() );

    super.dispose();
  }

  public static create<T extends Node[]>( ancestorNode: Node, nodes: T, layoutCallback: LayoutCallback<T> ): RelaxedManualConstraint<T> {
    return new RelaxedManualConstraint( ancestorNode, nodes, layoutCallback );
  }
}

scenery.register( 'RelaxedManualConstraint', RelaxedManualConstraint );
