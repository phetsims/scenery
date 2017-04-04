// Copyright 2017, University of Colorado Boulder

/**
 * Listens to presses (down events), attaching a listener to the pointer when one occurs, so that a release (up/cancel
 * or interruption) can be recorded.
 *
 * This is the base type for both DragListener and FireListener, which contains the shared logic that would be needed
 * by both.
 *
 * TODO: unit tests
 *
 * TODO: add example usage
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Property = require( 'AXON/Property' );

  /**
   * @constructor
   *
   * @param {Object} [options] - See the constructor body (below) for documented options.
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

      // {string|null} - Sets the pointer cursor to this value when this listener is "pressed". This means that even
      // when the mouse moves out of the node after pressing down, it will still have this cursor (overriding the
      // cursor of whatever nodes the pointer may be over).
      pressCursor: 'pointer',

      // {Function|null} - Called as press( event: {Event} ) when this listener's node is pressed (typically
      // from a down event, but can be triggered by other handlers).
      press: null,

      // {Function|null} - Called as release() when this listener's node is released (pointer up/cancel or interrupt
      // when pressed).
      release: null,

      // {Function|null} - Called as drag( event: {Event} ) when this listener's node is dragged (move events
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

    assert && assert( typeof options.mouseButton === 'number' &&
                      options.mouseButton >= 0 &&
                      options.mouseButton % 1 === 0, 'mouseButton should be a non-negative integer' );
    assert && assert( options.pressCursor === null || typeof options.pressCursor === 'string',
      'pressCursor should either be a string or null' );
    assert && assert( options.press === null || typeof options.press === 'function',
      'The press callback, if provided, should be a function' );
    assert && assert( options.release === null || typeof options.release === 'function',
      'The release callback, if provided, should be a function' );
    assert && assert( options.drag === null || typeof options.drag === 'function',
      'The drag callback, if provided, should be a function' );
    assert && assert( options.isPressedProperty instanceof Property && options.isPressedProperty.value === false,
      'If a custom isPressedProperty is provided, it must be a Property that is false initially' );
    assert && assert( options.targetNode === null || options.targetNode instanceof Node,
      'If provided, targetNode should be a Node' );
    assert && assert( typeof options.attach === 'boolean', 'attach should be a boolean' );

    // @public {Property.<Boolean>} [read-only] - Whether this listener is currently in the 'pressed' state or not
    this.isPressedProperty = options.isPressedProperty;

    // @public {Pointer|null} [read-only] - The current pointer, or null when not pressed.
    this.pointer = null;

    // @public {Trail|null} [read-only] - The Trail for the press, with no descendant nodes past the currentTarget
    // or targetNode (if provided). Will be null when not pressed.
    this.pressedTrail = null;

    // @public {boolean} [read-only] - Whether the last press was interrupted. Will be valid until the next press.
    this.interrupted = false;

    // @private - Stored options, see options for documentation.
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
       * @public (scenery-internal)
       *
       * @param {Event} event
       */
      up: function( event ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener pointer up' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        assert && assert( event.pointer === self.pointer );

        self.release();

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      /**
       * Called with 'cancel' events from the pointer (part of the listener API)
       * @public (scenery-internal)
       *
       * @param {Event} event
       */
      cancel: function( event ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener pointer cancel' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        assert && assert( event.pointer === self.pointer );

        self.interrupt(); // will mark as interrupted and release()

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      /**
       * Called with 'move' events from the pointer (part of the listener API)
       * @public (scenery-internal)
       *
       * @param {Event} event
       */
      move: function( event ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener pointer move' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        assert && assert( event.pointer === self.pointer );

        self.drag( event );

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      },

      /**
       * Called when the pointer needs to interrupt its current listener (usually so another can be added).
       * @public (scenery-internal)
       */
      interrupt: function() {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener pointer interrupt' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        self.interrupt();

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
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
     * Called with 'down' events (part of the listener API).
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly. See the press method instead.
     *
     * @param {Event} event
     */
    down: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener down' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.press( event );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Returns whether a press can be started with a particular event.
     * @public
     *
     * @param {Event} event
     * @returns {boolean}
     */
    canPress: function( event ) {
      // If this listener is already involved in pressing something, we can't press something
      if ( this.isPressed ) {
        return false;
      }

      // Only let presses be started with the correct mouse button.
      if ( event.pointer.isMouse && event.domEvent.button !== this._mouseButton ) {
        return false;
      }

      // We can't attach to a pointer that is already attached.
      if ( this._attach && event.pointer.isAttached() ) {
        return false;
      }

      return true;
    },

    /**
     * Moves the listener to the 'pressed' state if possible (attaches listeners and initializes press-related
     * properties).
     * @public
     *
     * This can be overridden (with super-calls) when custom press behavior is needed for a type.
     *
     * This can be called by outside clients in order to try to begin a process (generally on an already-pressed
     * pointer), and is useful if a 'drag' needs to change between listeners. Use canPress( event ) to determine if
     * a press can be started (if needed beforehand).
     *
     * @param {Event} event
     * @returns {boolean} success - Returns whether the press was actually started
     */
    press: function( event ) {
      assert && assert( event, 'An event is required' );

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener press' );

      if ( !this.canPress( event ) ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener could not press' );
        return false;
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener successful press' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // Set self properties before the property change, so they are visible to listeners.
      this.pointer = event.pointer;
      this.pressedTrail = this._targetNode ? this._targetNode.getUniqueTrail() :
                                             event.trail.subtrailTo( event.currentTarget, false );
      this.interrupted = false; // clears the flag (don't set to false before here)

      this.isPressedProperty.value = true;

      this.pointer.addInputListener( this._pointerListener, this._attach );
      this._listeningToPointer = true;

      this.pointer.cursor = this._pressCursor;

      // Notify after everything else is set up
      this._pressListener && this._pressListener( event );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();

      return true;
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
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener release' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      assert && assert( this.isPressed, 'This listener is not pressed' );

      this.isPressedProperty.value = false;

      this.pointer.removeInputListener( this._pointerListener );
      this._listeningToPointer = false;

      this.pointer.cursor = null;

      // Unset self properties after the property change, so they are visible to listeners beforehand.
      this.pointer = null;
      this.pressedTrail = null;

      // Notify after the rest of release is called in order to prevent it from triggering interrupt().
      // TODO: Is this a problem that we can't access things like this.pointer here?
      this._releaseListener && this._releaseListener();

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Called when move events are fired on the attached pointer listener.
     * @protected
     *
     * This can be overridden (with super-calls) when custom drag behavior is needed for a type.
     *
     * @param {Event} event
     */
    drag: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener drag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      assert && assert( this.isPressed, 'Can only drag while pressed' );

      this._dragListener && this._dragListener( event );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Interrupts the listener, releasing it (canceling behavior).
     * @public
     *
     * This can be called manually, but can also be called through node.interruptSubtreeInput().
     */
    interrupt: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener interrupt' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      if ( this.isPressed ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener interrupting' );
        this.interrupted = true;

        this.release();
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Disposes the listener, releasing references. It should not be used after this.
     * @public
     */
    dispose: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener dispose' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      if ( this._listeningToPointer ) {
        this.pointer.removeInputListener( this._pointerListener );
      }

      // TODO: Should we dispose our properties like isPressedProperty? If so, we'll have to be more careful with
      // multilinks, and there will be more overhead.

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }
  } );

  return PressListener;
} );
