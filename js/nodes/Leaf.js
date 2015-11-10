// Copyright 2015, University of Colorado Boulder

/**
 * A mixin for subtypes of Node, used to prevent children being added/removed to that subtype of Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  var Leaf = {
    mixin: function( type ) {
      var proto = type.prototype;

      proto.insertChild = function( index, node ) {
        throw new Error( 'Attempt to insert child into Leaf' );
      };

      proto.removeChildWithIndex = function( node, indexOfChild ) {
        throw new Error( 'Attempt to remove child from Leaf' );
      };
    }
  };
  scenery.register( 'Leaf', Leaf );

  return scenery.Leaf;
} );
