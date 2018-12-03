// Copyright 2018, University of Colorado Boulder

/**
 * Tracks the state of accessible focus.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

define( require => {
  'use strict';

  var AccessibleInstance = require( 'SCENERY/accessibility/AccessibleInstance' );
  var Pointer = require( 'SCENERY/input/Pointer' ); // inherits from Pointer
  var scenery = require( 'SCENERY/scenery' );
  var Trail = require( 'SCENERY/util/Trail' );
  // var Display = require( 'SCENERY/display/Display' ); // so requireJS doesn't balk about circular dependency
  var Focus = require( 'SCENERY/accessibility/Focus' );

  class A11yPointer extends Pointer {
    constructor() {
      super( null, false );

      this.type = 'a11y';

      sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Created ' + this.toString() );
    }

    /**
     * Set up listeners, attaching blur and focus listeners to the pointer once this A11yPointer has been attached
     * to a display.
     * @private (scenery-internal)
     * 
     * @param  {Display} display
     */
    initializeListeners( display ) {
      this.addInputListener( {
        focus: () => {
          assert && assert( this.trail, 'trail should have been calculated for the focused node' );

          // NOTE: The "root" peer can't be focused (so it doesn't matter if it doesn't have a node).
          if ( this.trail.lastNode().focusable ) {
            scenery.Display.focus = new Focus( display, AccessibleInstance.guessVisualTrail( this.trail, display.rootNode ) );
            display.pointerFocus = null;
          }
        },
        blur: ( event ) => {
          scenery.Display.focus = null;
        }
      } );
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