// Copyright 2018-2021, University of Colorado Boulder

/**
 * A Property wrapper for Node TinyProperty/Property instances.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import Property from '../../../axon/js/Property.js';
// For TypeScript support
import { scenery, Node } from '../imports.js';  // eslint-disable-line no-unused-vars

class NodeProperty extends Property {

  /**
   * @param {Node} node
   * @param {TinyProperty|Property} property
   * @param {string} attribute - for example 'left' or 'centerBottom' or 'visible' or 'pickable'
   * @param {Object} [options]
   */
  constructor( node, property, attribute, options ) {
    assert && assert( typeof attribute === 'string', 'wrong type for getLocation' );

    // Read-only Property that describes a part relative to the bounds of the node.
    super( node[ attribute ], options );

    // @private {Node}
    this.node = node;

    // @private {TinyProperty|Property}
    this.property = property;

    // @private {string}
    this.attribute = attribute;

    // @private {function} - When the node event is triggered, get the new value and set it to this Property
    this.changeListener = () => this.set( node[ attribute ] );

    this.property.lazyLink( this.changeListener );
  }

  /**
   * Unlinks listeners when disposed.  Must be called before the corresponding Node is disposed.
   * @public
   */
  dispose() {
    assert && assert( !this.node.isDisposed, 'NodeProperty should be disposed before corresponding Node is disposed' );
    this.property.unlink( this.changeListener );
    super.dispose();
    this.node = null;
  }

  /**
   * Updates the value of this node, overridden to update the Node value
   * @override
   * @param {*} value - the new value this Property will take, which is different than the previous value.
   * @protected - can be overridden.
   */
  setPropertyValue( value ) {

    // Set the node value first, as if it was the first link listener.
    this.node[ this.attribute ] = value;

    super.setPropertyValue( value );
  }
}

scenery.register( 'NodeProperty', NodeProperty );
export default NodeProperty;