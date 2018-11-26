// Copyright 2018, University of Colorado Boulder

/**
 * Tracks the state of accessible focus.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

define( require => {
  'use strict';

  var Pointer = require( 'SCENERY/input/Pointer' ); // inherits from Pointer
  var scenery = require( 'SCENERY/scenery' );
  var Trail = require( 'SCENERY/util/Trail' );

  class A11yPointer extends Pointer {
    constructor() {
      super( null, false );

      this.type = 'a11y';


      sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Created ' + this.toString() );
    }

    /**
     * @param {Node} rootNode
     * @param {string} trailId
     * @public
     * @returns {Trail} - updated trail
     */
    updateTrail( rootNode, trailId ) {
      if ( this.trail && this.trail.getUniqueId() === trailId ) {
        return this.trail;
      }
      var trail = Trail.fromUniqueId( rootNode, trailId );
      this.trail = trail;
      return trail;
    }
  }

  return scenery.register( 'A11yPointer', A11yPointer );
} );