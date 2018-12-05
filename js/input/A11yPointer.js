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

    /**
     * Pointer type for managing accessibility, in particular the focus in the display
     * @param {Display} display
     */
    constructor( display ) {
      super( null, false );

      // @public
      this.type = 'a11y';

      // @private
      this.display = display;

      this.initializeListeners();

      sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Created ' + this.toString() );
    }

    /**
     * Set up listeners, attaching blur and focus listeners to the pointer once this A11yPointer has been attached
     * to a display.
     * @private
     */
    initializeListeners() {
      this.addInputListener( {
        focus: () => {
          assert && assert( this.trail, 'trail should have been calculated for the focused node' );

          // NOTE: The "root" peer can't be focused (so it doesn't matter if it doesn't have a node).
          if ( this.trail.lastNode().focusable ) {
            scenery.Display.focus = new Focus( this.display, AccessibleInstance.guessVisualTrail( this.trail, this.display.rootNode ) );
            this.display.pointerFocus = null;
          }
        },
        blur: () => {
          scenery.Display.focus = null;
        }
      } );
    }

    /**
     * @param {string} trailId
     * @public
     * @returns {Trail} - updated trail
     */
    updateTrail( trailId ) {
      if ( this.trail && this.trail.getUniqueId() === trailId ) {
        return this.trail;
      }
      let trail = Trail.fromUniqueId( this.display.rootNode, trailId );
      this.trail = trail;
      return trail;
    }

    /**
     * Assert that the given trail matches the one stored by this pointer.
     * @param {string} trailString
     */
    invalidateTrail( trailString ) {

      // The trail is set to null on blur, and we can't guarantee that events will always come in order
      // (i.e. setTimeout in KeyboardFuzzer)
      // This is more inefficient than just asserting out, it would be good to know how often we have to
      // recalculate this way.
      if ( this.trail === null || this.trail.uniqueId !== trailString ) {
        this.trail = Trail.fromUniqueId( this.display.rootNode, trailString );
      }
    }
  }

  return scenery.register( 'A11yPointer', A11yPointer );
} );