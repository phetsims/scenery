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
   * @param {Node} rootNode
   * @param {Array.<Node>} nodes
   * @param {function(LayoutProxy+)} layoutCallback
   * @param {Object} [options]
   */
  constructor( rootNode, nodes, layoutCallback, options ) {

    assert && assert( rootNode instanceof Node );
    assert && assert( Array.isArray( nodes ) && _.every( nodes, node => node instanceof Node ) );
    assert && assert( typeof layoutCallback === 'function' );

    options = merge( {

    }, options );

    super( rootNode );

    // @private
    this.rootNode = rootNode;
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
   * @param {Node} rootNode
   * @param {Array.<Node>} nodes
   * @param {function(LayoutProxy+)} layoutCallback
   * @param {Object} [options]
   * @returns {ManualConstraint}
   */
  static create( rootNode, nodes, layoutCallback, options ) {
    return new ManualConstraint( rootNode, nodes, layoutCallback, options );
  }
}

scenery.register( 'ManualConstraint', ManualConstraint );

export default ManualConstraint;