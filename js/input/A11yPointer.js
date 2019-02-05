// Copyright 2018, University of Colorado Boulder

/**
 * Pointer type for managing accessibility, in particular the focus in the display.
 * Tracks the state of accessible focus.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

define( require => {
  'use strict';

  const AccessibleInstance = require( 'SCENERY/accessibility/AccessibleInstance' );
  const Pointer = require( 'SCENERY/input/Pointer' ); // inherits from Pointer
  const scenery = require( 'SCENERY/scenery' );
  const Trail = require( 'SCENERY/util/Trail' );
  // const Display = require( 'SCENERY/display/Display' ); // so requireJS doesn't balk about circular dependency
  const Focus = require( 'SCENERY/accessibility/Focus' );

  class A11yPointer extends Pointer {

    /**
     * @param {Display} display
     */
    constructor( display ) {
      super( null, false, 'a11y' );

      // @private
      this.display = display;

      this.initializeListeners();

      // @public (scenery-internal) - Prevent any "trusted" events from being dispatched to the KeyStateTracker. When
      // true, only scripted events are passed to the keyStateTracker. Otherwise, the modeled keyboard state when using
      // fuzzBoard will appear broken as both user and KeyboardFuzzer interact with display.
      this.blockTrustedEvents = false;

      // @private {Node|null} - target of a user event, if focus changes in response to keydown listeners, listeners
      // on keyup are prevented because the key press was not intended for the newly focused node.
      // TODO: Can we do this for more than keydown/keyup? See https://github.com/phetsims/scenery/issues/942
      this.keydownTargetNode = null; 

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
        blur: ( event ) => {
          scenery.Display.focus = null;
          this.keydownTargetNode = null;
        },
        keydown: ( event ) => {
          if ( this.blockTrustedEvents && event.domEvent.isTrusted ) {
            return;
          }
          scenery.Display.keyStateTracker.keydownUpdate( event );

          // set the target to potentially block keyup events
          this.keydownTargetNode = event.target;
        },
        keyup: ( event ) => {
          if ( this.blockTrustedEvents && event.domEvent.isTrusted ) {
            return;
          }
          scenery.Display.keyStateTracker.keyupUpdate( event );

          // The keyup event was received on a node that didn't receive a keydown event, abort to prevent any other
          // listeners from being called for this event. Done after updating KeyStateTracker so that the global state
          // of the keyboard is still accurate
          if ( this.keydownTargetNode !== event.target ) {
            event.abort();
          }
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
      const trail = Trail.fromUniqueId( this.display.rootNode, trailId );
      this.trail = trail;
      return trail;
    }

    /**
     * Recompute the trail to the node under this A11yPointer. Updating the trail here is generally not necessary since
     * it is recomputed on focus. But there are times where a11y events can be called out of order with focus/blur
     * and the trail will either be null or stale. This might happen more often when scripting fake browser events
     * with a timeout (like in fuzzBoard).
     *
     * @public (scenery-internal)
     * @param {string} trailString
     */
    invalidateTrail( trailString ) {
      if ( this.trail === null || this.trail.uniqueId !== trailString ) {
        this.trail = Trail.fromUniqueId( this.display.rootNode, trailString );
      }
    }
  }

  return scenery.register( 'A11yPointer', A11yPointer );
} );