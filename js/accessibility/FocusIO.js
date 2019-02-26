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
  var validate = require( 'AXON/validate' );

  /**
   * @param {Focus} focus - the focus region which has {display,trail}
   * @param {string} phetioID - the unique tandem assigned to the focus
   * @constructor
   */
  function FocusIO( focus, phetioID ) {
    ObjectIO.call( this, focus, phetioID );
  }

  phetioInherit( ObjectIO, 'FocusIO', FocusIO, {}, {
    validator: { valueType: Focus }, // TODO: Should this support null?

    /**
     * Convert the focus region to a plain JS object for serialization.
     * @param {Object} focus - the focus region which has {display,trail}
     * @returns {Object} - the serialized object
     * @override
     */
    toStateObject: function( focus ) {

      // If nothing is focused, the focus is null
      if ( focus === null ) {
        return null;
      }
      else {
        validate( focus, this.validator );
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
          focusedPhetioID: focus.trail.lastNode().tandem.phetioID
        };
      }
    },

    documentation: 'A IO type for the instance in the simulation which currently has keyboard focus.'
  } );

  scenery.register( 'FocusIO', FocusIO );

  return FocusIO;
} );