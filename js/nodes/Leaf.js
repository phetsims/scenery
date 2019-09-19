// Copyright 2015-2019, University of Colorado Boulder

/**
 * A trait for subtypes of Node, used to prevent children being added/removed to that subtype of Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const inheritance = require( 'PHET_CORE/inheritance' );
  const Node = require( 'SCENERY/nodes/Node' );
  const scenery = require( 'SCENERY/scenery' );

  var Leaf = {
    /**
     * Removes the capability to insert children when this is mixed into a type.
     * @public
     * @trait
     *
     * @param {function} type - The type (constructor) whose prototype we'll modify so that it can't have children.
     */
    mixInto: function( type ) {
      assert && assert( _.includes( inheritance( type ), Node ) );

      var proto = type.prototype;

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

  return scenery.Leaf;
} );
