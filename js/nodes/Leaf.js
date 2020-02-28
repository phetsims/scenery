// Copyright 2015-2020, University of Colorado Boulder

/**
 * A trait for subtypes of Node, used to prevent children being added/removed to that subtype of Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import inheritance from '../../../phet-core/js/inheritance.js';
import scenery from '../scenery.js';
import Node from './Node.js';

const Leaf = {
  /**
   * Removes the capability to insert children when this is mixed into a type.
   * @public
   * @trait {Node}
   *
   * @param {function} type - The type (constructor) whose prototype we'll modify so that it can't have children.
   */
  mixInto: function( type ) {
    assert && assert( _.includes( inheritance( type ), Node ) );

    const proto = type.prototype;

    /**
     * @override
     */
    proto.insertChild = function( index, node ) {
      throw new Error( 'Attempt to insert child into Leaf' );
    };

    /**
     * @override
     */
    proto.removeChildWithIndex = function( node, indexOfChild ) {
      throw new Error( 'Attempt to remove child from Leaf' );
    };
  }
};
scenery.register( 'Leaf', Leaf );

export default scenery.Leaf;