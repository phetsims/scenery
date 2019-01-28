// Copyright 2013-2016, University of Colorado Boulder

/**
 * A Scenery Event is an abstraction over incoming user DOM events.
 *
 * It provides more information (particularly Scenery-related information), and handles a single pointer at a time
 * (DOM TouchEvents can include information for multiple touches at the same time, so the TouchEvent can be passed to
 * multiple Scenery events). Thus it is not save to assume that the DOM event is unique, as it may be shared.
 *
 * NOTE: While the event is being dispatched, its currentTarget may be changed. It is not fully immutable.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const A11yPointer = require( 'SCENERY/input/A11yPointer' );
  const Mouse = require( 'SCENERY/input/Mouse' );
  const Pointer = require( 'SCENERY/input/Pointer' );
  const scenery = require( 'SCENERY/scenery' );
  const Trail = require( 'SCENERY/util/Trail' );

  /**
   * @param {Trail} trail - The trail to the node picked/hit by this input event.
   * @param {string} type - Type of the event, e.g. 'string'
   * @param {Pointer} pointer - The pointer that triggered this event
   * @param {DOMEvent|null} domEvent - The original DOM Event that caused this Event to fire.
   */
  class Event {
    constructor( trail, type, pointer, domEvent ) {
      assert && assert( trail instanceof Trail, 'Event\'s trail parameter should be a {Trail}' );
      assert && assert( typeof type === 'string', 'Event\'s type should be a {string}' );
      assert && assert( pointer instanceof Pointer, 'Event\'s pointer parameter should be a {Pointer}' );
      // TODO: add domEvent type assertion -- will browsers support this?

      // @public (read-only) {boolean} - Whether this Event has been 'handled'. If so, it will not bubble further.
      this.handled = false;

      // @public (read-only) {boolean} - Whether this Event has been 'aborted'. If so, no further listeners with it will fire.
      this.aborted = false;

      // @public {Trail} - Path to the leaf-most node "hit" by the event, ordered list, from root to leaf
      this.trail = trail;

      // @public {string} - What event was triggered on the listener, e.g. 'move'
      this.type = type;

      // @public {Pointer} - The pointer that triggered this event
      this.pointer = pointer;

      // @public {DOMEvent|null} - Raw DOM InputEvent (TouchEvent, PointerEvent, MouseEvent,...)
      this.domEvent = domEvent;

      // @public {Node|null} - whatever node you attached the listener to, or null when firing events on a Pointer
      this.currentTarget = null;

      // @public {Node} - Leaf-most node in trail
      this.target = trail.lastNode();

      // @public {boolean} - Whether this is the 'primary' mode for the pointer. Always true for touches, and will be true
      // for the mouse if it is the primary (left) mouse button.
      // TODO: don't require check on domEvent (seems sometimes this is passed as null as a hack?)
      this.isPrimary = !( pointer instanceof Mouse ) || !domEvent || domEvent.button === 0;

      // Store the last-used non-null DOM event for future use if required.
      if ( domEvent ) {
        pointer.lastDOMEvent = domEvent;
      }
    }

    /**
     * like DOM Event.stopPropagation(), but named differently to indicate it doesn't fire that behavior on the underlying DOM event
     * @public
     */
    handle() {
      sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'handled event' );
      this.handled = true;
    }

    /**
     * like DOM Event.stopImmediatePropagation(), but named differently to indicate it doesn't fire that behavior on the underlying DOM event
     * @public
     */
    abort() {
      sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'aborted event' );
      this.aborted = true;
    }

    /**
     * Specifies whether or not the Event came from alternative input. See Input.A11Y_EVENT_TYPES for a list of events
     * a11y related events supported by scenery.
     * @public
     * @returns {boolean}
     */
    isA11y() {
      return this.pointer instanceof A11yPointer;
    }

    /**
     * Returns whether a typical PressListener (that isn't already attached) could start a drag with this event.
     * @public
     *
     * This can typically be used for patterns where no action should be taken if a press can't be started, e.g.:
     *
     *   down: function( event ) {
     *     if ( !event.canStartPress() ) { return; }
     *
     *     // ... Do stuff to create a node with some type of PressListener
     *
     *     dragListener.press( event );
     *   }
     *
     * NOTE: This ignores non-left mouse buttons (as this is the typical behavior). Custom checks should be done if this
     *       is not suitable.
     *
     * @returns {boolean}
     */
    canStartPress() {
      // If the pointer is already attached (some other press probably), it can't start a press.
      // Additionally, we generally want to ignore non-left mouse buttons.
      return !this.pointer.isAttached() && ( !( this.pointer instanceof Mouse ) || this.domEvent.button === 0 );
    }
  }

  return scenery.register( 'Event', Event );
} );
