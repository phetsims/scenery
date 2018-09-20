// Copyright 2015-2016, University of Colorado Boulder

/**
 * A trait for subtypes of Node, used to prevent children being added/removed to that subtype of Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inheritance = require( 'PHET_CORE/inheritance' );
  var Node = require( 'SCENERY/nodes/Node' );
  var scenery = require( 'SCENERY/scenery' );

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
