// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import Constraint from './Constraint.js';

class ManualConstraint extends Constraint {
  /**
   * @param {Node} ancestorNode
   * @param {Array.<Node>} nodes
   * @param {function(LayoutProxy+)} layoutCallback
   * @param {Object} [options]
   */
  constructor( ancestorNode, nodes, layoutCallback, options ) {

    assert && assert( ancestorNode instanceof Node );
    assert && assert( Array.isArray( nodes ) && _.every( nodes, node => node instanceof Node ) );
    assert && assert( typeof layoutCallback === 'function' );

    options = merge( {

    }, options );

    super( ancestorNode );

    // @private
    this.ancestorNode = ancestorNode;
    this.nodes = nodes;
    this.layoutCallback = layoutCallback;

    // @private {function} - Minimizing garbage created
    this.proxyFactory = this.createLayoutProxy.bind( this );

    // Hook up to listen to these nodes
    this.nodes.forEach( node => this.addNode( node ) );

    // Run the layout manually at the start
    this.updateLayout();
  }

  /**
   * @protected
   * @override
   */
  layout() {
    super.layout();

    assert && assert( _.every( this.nodes, node => !node.isDisposed ) );

    const proxies = this.nodes.map( this.proxyFactory );

    this.layoutCallback.apply( null, proxies );

    // Minimizing garbage created
    for ( let i = 0; i < proxies.length; i++ ) {
      proxies[ i ].dispose();
    }
  }

  /**
   * @public
   *
   * @param {Node} ancestorNode
   * @param {Array.<Node>} nodes
   * @param {function(LayoutProxy+)} layoutCallback
   * @param {Object} [options]
   * @returns {ManualConstraint}
   */
  static create( ancestorNode, nodes, layoutCallback, options ) {
    return new ManualConstraint( ancestorNode, nodes, layoutCallback, options );
  }
}

scenery.register( 'ManualConstraint', ManualConstraint );

export default ManualConstraint;