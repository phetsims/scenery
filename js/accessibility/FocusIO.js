// Copyright 2016, University of Colorado Boulder

/**
 * IO type for Focus
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var Focus = require( 'SCENERY/accessibility/Focus' );
  var ObjectIO = require( 'TANDEM/types/ObjectIO' );
  var phetioInherit = require( 'TANDEM/phetioInherit' );
  var scenery = require( 'SCENERY/scenery' );

  // ifphetio
  var assertInstanceOf = require( 'ifphetio!PHET_IO/assertInstanceOf' );

  /**
   * @param {Focus} focus - the focus region which has {display,trail}
   * @param {string} phetioID - the unique tandem assigned to the focus
   * @constructor
   */
  function FocusIO( focus, phetioID ) {
    assert && assertInstanceOf( focus, Focus );
    ObjectIO.call( this, focus, phetioID );
  }

  phetioInherit( ObjectIO, 'FocusIO', FocusIO, {}, {

    /**
     * Convert the focus region to a plain JS object for serialization.
     * @param {Object} focus - the focus region which has {display,trail}
     * @returns {Object} - the serialized object
     * @override
     */
    toStateObject: function( focus ) {

      // If nothing is focused, the focus is nulls
      if ( focus === null ) {
        return null;
      }
      else {
        assert && assertInstanceOf( focus, Focus );
        var phetioIDIndices = [];
        focus.trail.nodes.forEach( function( node, i ) {

          // Don't include the last node, since it is the focused node
          if ( i < focus.trail.nodes.length - 1 ) {

            // If the node was PhET-iO instrumented, include its phetioID instead of its index (because phetioID is more stable)
            if ( node.tandem ) {
              phetioIDIndices.push( node.tandem.phetioID );
            }
            else {
              phetioIDIndices.push( focus.trail.indices[ i ] );
            }
          }
        } );

        return {
          focusedPhetioID: focus.trail.lastNode().tandem.phetioID,
          indices: focus.trail.indices,
          phetioIDIndices: phetioIDIndices
        };
      }
    },

    /**
     * Convert the serialized instance back to a focus object
     * @param {Object} stateObject
     * @returns {Object} with {display,trail}
     * @override
     */
    fromStateObject: function( stateObject ) {

      if ( stateObject === null ) {

        // support unfocused
        return null;
      }
      else {
        var indices = stateObject.indices;

        // Follow the path of children based on their indices, starting from the root of the display.
        // There is always one more node in Trail than indices, representing the root node.
        var currentNode = phet.joist.sim.display.rootNode;
        var nodes = [ currentNode ];
        for ( var i = 0; i < indices.length; i++ ) {
          var index = indices[ i ];
          currentNode = currentNode.children[ index ];
          nodes.push( currentNode );
        }

        return { display: phet.joist.sim.display, trail: new phet.scenery.Trail( nodes ) };
      }
    },

    documentation: 'A IO type for the instance in the simulation which currently has keyboard focus.'
  } );

  scenery.register( 'FocusIO', FocusIO );

  return FocusIO;
} );