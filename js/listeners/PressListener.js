// Copyright 2013-2017, University of Colorado Boulder

/**
 * Listens to presses (down events), attaching a listener to the pointer when one occurs, so that a release (up/cancel
 * or interruption) can be recorded.
 *
 * TODO: unit tests
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Property = require( 'AXON/Property' );

  /**
   * @constructor
   *
   * @params {Object} [options] - See the constructor body (below) for documented options.
   */
  function PressListener( options ) {
    var self = this;

    options = _.extend( {
      // {number} - Restricts to the specific mouse button (but allows any touch). Only one mouse button is allowed at
      // a time. The button numbers are defined in https://developer.mozilla.org/en-US/docs/Web/API/MouseEvent/button,
      // where typically:
      //   0: Left mouse button
      //   1: Middle mouse button (or wheel press)
      //   2: Right mouse button
      //   3+: other specific numbered buttons that are more rare
      mouseButton: 0,

      // {string} - Sets the pointer cursor to this value when this listener is "pressed". This means that even when
      // the mouse moves out of the node after pressing down, it will still have this cursor (overriding the cursor of
      // whatever nodes the pointer may be over).
      pressCursor: 'pointer',

      // {Function|null} - Called as press( event: {scenery.Event} ) when this listener's node is pressed (typically
      // from a down event, but can be triggered by other handlers).
      press: null,

      // {Function|null} - Called as release() when this listener's node is released (pointer up/cancel or interrupt
      // when pressed).
      release: null,

      // TODO: decide on {scenery.Event} or {Event} docs, as window.Event is the DOM event type (also potentially
      //       documented as {Event}
      // {Function|null} - Called as drag( event: {scenery.Event} ) when this listener's node is dragged (move events
      // on the pointer while pressed).
      drag: null,

      // {Property.<Boolean>} - If provided, this property will be used to track whether this listener's node is
      // "pressed" or not.
      isPressedProperty: new Property( false ),

      // {Node|null} - If provided, the pressedTrail (calculated from the down event) will be replaced with
      targetNode: null,

      // {boolean} - If true, this listener will not "press" while the associated pointer is attached, and when pressed,
      // will mark itself as attached to the pointer. If this listener should not be interrupted by others and isn't
      // a "primary" handler of the pointer's behavior, this should be set to false.
      attach: true
    }, options );

    assert && assert( options.isPressedProperty.value === false,
      'If a custom isPressedProperty is provided, it must be false initially' );

    // @public {Property.<Boolean>} [read-only] - Whether this listener is currently in the 'pressed' state or not
    this.isPressedProperty = options.isPressedProperty;

    // @public {Pointer|null} [read-only] - The current pointer (if pressed)
    this.pointer = null;

    // @public {Trail|null} [read-only] - The Trail for the press, with no descendant nodes past the currentTarget
    this.pressedTrail = null;

    // @public {boolean} [read-only] - Whether the last press was interrupted. Will be valid until the next press.
    this.interrupted = false;

    // @private (stored options)
    this._mouseButton = options.mouseButton;
    this._pressCursor = options.pressCursor;
    this._pressListener = options.press;
    this._releaseListener = options.release;
    this._dragListener = options.drag;
    this._targetNode = options.targetNode;
    this._attach = options.attach;

    // @private {boolean} - Whether our pointer listener is referenced by the pointer (need to have a flag due to
    //                      handling disposal properly).
    this._listeningToPointer = false;

    // @private {Object} - The listener that gets added to the pointer when we are pressed
    this._pointerListener = {
      /**
       * Called with 'up' events from the pointer (part of the listener API)
       * @public
       *
       * @param {Event} event
       */
      up: function( event ) {
        assert && assert( event.pointer === self.pointer );

        self.release();
      },

      /**
       * Called with 'cancel' events from the pointer (part of the listener API)
       * @public
       *
       * @param {Event} event
       */
      cancel: function( event ) {
        assert && assert( event.pointer === self.pointer );

        self.interrupt(); // will mark as interrupted and release()
      },

      /**
       * Called with 'move' events from the pointer (part of the listener API)
       * @public
       *
       * @param {Event} event
       */
      move: function( event ) {
        assert && assert( event.pointer === self.pointer );

        self.drag( event );
      },

      /**
       * Called when the pointer needs to interrupt its current listener (usually so another can be added).
       * @public
       */
      interrupt: function() {
        self.interrupt();
      }
    };
  }

  scenery.register( 'PressListener', PressListener );

  inherit( Object, PressListener, {
    /**
     * Whether this listener is currently activated with a press.
     * @public
     *
     * @returns {boolean}
     */
    get isPressed() {
      return this.isPressedProperty.value;
    },

    /**
     * The main node that this listener is responsible for dragging.
     * @public
     *
     * @returns {Node}
     */
    getCurrentTarget: function() {
      assert && assert( this.isPressed, 'We have no currentTarget if we are not pressed' );

      return this.pressedTrail.lastNode();
    },

    /**
     * Called with 'down' events from the pointer (part of the listener API).
     * @public
     *
     * NOTE: Do not call directly. See the press method instead.
     *
     * @param {Event} event
     */
    down: function( event ) {
      this.press( event );
    },

    /**
     * Moves the listener to the 'pressed' state if possible (attaches listeners and initializes press-related
     * properties).
     * @public
     *
     * This can be overridden (with super-calls) when custom press behavior is needed for a type.
     *
     * This can be called by outside clients in order to try to begin a process (generally on an already-pressed
     * pointer), and is useful if a 'drag' needs to change between listeners.
     *
     * TODO: Can we separate this out so it's possible to not have to pass in the original event?
     *       The pointer, trail and currentTarget should be sufficient (if we don't need to pass it to our listener)
     *       Pointer is sufficient if a targetNode is provided.
     *
     * @param {Event} event
     */
    press: function( event ) {
      assert && assert( !this.isPressed, 'This listener is already pressed' );

      // If this listener is already involved in pressing something, we can't press something
      if ( this.isPressed ) { return; }

      // Only let presses be started with the correct mouse button.
      if ( event.pointer.isMouse && event.domEvent.button !== this._mouseButton ) { return; }

      // We can't attach to a pointer that is already attached.
      if ( this._attach && event.pointer.isAttached() ) { return; }

      // Set self properties before the property change, so they are visible to listeners.
      this.pointer = event.pointer;
      this.pressedTrail = this._targetNode ? this._targetNode.getUniqueTrail() :
                                             event.trail.subtrailTo( event.currentTarget, false );
      this.interrupted = false; // clears the flag (don't set to false before here)

      this.isPressedProperty.value = true;

      this.pointer.addInputListener( this._pointerListener, this._attach );
      this._listeningToPointer = true;

      this.pointer.cursor = this._pressCursor;

      this._pressListener && this._pressListener( event );
    },

    /**
     * Releases a pressed listener.
     * @public
     *
     * This can be overridden (with super-calls) when custom release behavior is needed for a type.
     *
     * This can be called from the outside to release the press without the pointer having actually fired any 'up'
     * events. If the cancel/interrupt behavior is more preferable, call interrupt() on this listener instead.
     */
    release: function() {
      assert && assert( this.isPressed, 'This listener is not pressed' );

      this.isPressedProperty.value = false;

      this.pointer.removeInputListener( this._pointerListener );
      this._listeningToPointer = false;

      this.pointer.cursor = null;

      // Unset self properties after the property change, so they are visible to listeners beforehand.
      this.pointer = null;
      this.pressedTrail = null;

      this._releaseListener && this._releaseListener();
    },

    /**
     * Called when move events are fired on the attached pointer listener.
     * @protected
     *
     * This can be overridden (with super-calls) when custom drag behavior is needed for a type.
     */
    drag: function( event ) {
      assert && assert( this.isPressed, 'Can only drag while pressed' );

      this._dragListener && this._dragListener( event );
    },

    /**
     * Interrupts the listener, releasing it (canceling behavior).
     * @public
     *
     * This can be called manually, but can also be called through node.interruptSubtreeInput().
     */
    interrupt: function() {
      if ( this.isPressed ) {
        this.interrupted = true;

        this.release();
      }
    },

    /**
     * Disposes the listener, releasing references. It should not be used after this.
     * @public
     */
    dispose: function() {
      if ( this._listeningToPointer ) {
        this.pointer.removeInputListener( this._pointerListener );
      }
    }
  } );

  return PressListener;
} );
