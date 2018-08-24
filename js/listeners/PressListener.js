// Copyright 2017, University of Colorado Boulder

/**
 * Listens to presses (down events), attaching a listener to the pointer when one occurs, so that a release (up/cancel
 * or interruption) can be recorded.
 *
 * This is the base type for both DragListener and FireListener, which contains the shared logic that would be needed
 * by both.
 *
 * For example usage, see scenery/examples/input.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var BooleanProperty = require( 'AXON/BooleanProperty' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Mouse = require( 'SCENERY/input/Mouse' );
  var Node = require( 'SCENERY/nodes/Node' );
  var ObservableArray = require( 'AXON/ObservableArray' );
  var PhetioObject = require( 'TANDEM/PhetioObject' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );
  var Tandem = require( 'TANDEM/Tandem' );
  var Timer = require( 'PHET_CORE/Timer' );

  // ifphetio
  var PressListenerIO = require( 'SCENERY/listeners/PressListenerIO' );

  // global
  var globalID = 0;

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

      // {Function|null} - Called as press( event: {Event}, listener: {PressListener} ) when this listener's node is
      // pressed (typically from a down event, but can be triggered by other handlers).
      press: null,

      // {Function|null} - Called as release( listener: {PressListener} ) when this listener's node is released
      // (pointer up/cancel or interrupt when pressed).
      release: null,

      // {Function|null} - Called as drag( event: {Event}, listener: {PressListener} ) when this listener's node is
      //dragged (move events on the pointer while pressed).
      drag: null,

      // {Property.<Boolean>} - If provided, this property will be used to track whether this listener's node is
      // "pressed" or not.
      isPressedProperty: new BooleanProperty( false, { reentrant: true } ),

      // {Property.<boolean>} - A property that will be controlled by this listener. It will be set to true when at
      // least one pointer is over the listener.
      // A custom property may be passed in here, as it may be useful for hooking up to existing button models.
      isOverProperty: new Property( false ),

      // {Property.<boolean>} - A property that will be controlled by this listener. It will be set to true when either:
      //   1. The listener is pressed and the pointer that is pressing is over the listener.
      //   2. There is at least one unpressed pointer that is over the listener.
      // A custom property may be passed in here, as it may be useful for hooking up to existing listener models.
      isHoveringProperty: new Property( false ),

      // {Property.<boolean>} - A property that will be controlled by this listener. It will be set to true when either:
      //   1. The listener is pressed.
      //   2. There is at least one unpressed pointer that is over the listener.
      // This is essentially true when ( isPressed || isHovering ).
      // A custom property may be passed in here, as it may be useful for hooking up to existing listener models.
      isHighlightedProperty: new Property( false ),

      // {Node|null} - If provided, the pressedTrail (calculated from the down event) will be replaced with the
      // (sub)trail that ends with the targetNode as the leaf-most Node. This affects the parent coordinate frame
      // computations.
      targetNode: null,

      // {boolean} - If true, this listener will not "press" while the associated pointer is attached, and when pressed,
      // will mark itself as attached to the pointer. If this listener should not be interrupted by others and isn't
      // a "primary" handler of the pointer's behavior, this should be set to false.
      attach: true,

      // {function} - Checks this when trying to start a press. If this function returns false, a press will not be
      // started.
      canStartPress: _.constant( true ),

      // a11y - interval between a11y click presses. Same default as ButtonModel.js
      fireOnHoldInterval: 100,

      // {Tandem} - For instrumenting
      tandem: Tandem.required,

      // {PressListenerIO} - PhET-iO type
      phetioType: PressListenerIO,

      // {boolean}
      phetioState: false,

      // {string}
      phetioEventType: 'user'
    }, options );

    PhetioObject.call( this, options );

    assert && assert( typeof options.mouseButton === 'number' && options.mouseButton >= 0 && options.mouseButton % 1 === 0,
      'mouseButton should be a non-negative integer' );
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
    assert && assert( options.isOverProperty instanceof Property && options.isOverProperty.value === false,
      'If a custom isOverProperty is provided, it must be a Property that is false initially' );
    assert && assert( options.isHoveringProperty instanceof Property && options.isHoveringProperty.value === false,
      'If a custom isHoveringProperty is provided, it must be a Property that is false initially' );
    assert && assert( options.isHighlightedProperty instanceof Property && options.isHighlightedProperty.value === false,
      'If a custom isHighlightedProperty is provided, it must be a Property that is false initially' );

    // @private {number} - Unique global ID for this listener
    this._id = globalID++;

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' construction' );

    // @public {ObservableArray.<Pointer>} - Contains all pointers that are over our button. Tracked by adding with
    // 'enter' events and removing with 'exit' events.
    this.overPointers = new ObservableArray();

    // @public {Property.<Boolean>} [read-only] - See notes in options documentation
    this.isPressedProperty = options.isPressedProperty;
    this.isOverProperty = options.isOverProperty;
    this.isHoveringProperty = options.isHoveringProperty;
    this.isHighlightedProperty = options.isHighlightedProperty;

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
    this._canStartPress = options.canStartPress;
    this._fireOnHoldInterval = options.fireOnHoldInterval;

    // @private {boolean} - Whether our pointer listener is referenced by the pointer (need to have a flag due to
    //                      handling disposal properly).
    this._listeningToPointer = false;

    // @private {function} - isHoveringProperty updates (not a DerivedProperty because we need to hook to passed-in
    // properties)
    this._isHoveringListener = this.invalidateHovering.bind( this );

    // @private {function} - isHighlightedProperty updates (not a DerivedProperty because we need to hook to passed-in
    // properties)
    this._isHighlightedListener = this.invalidateHighlighted.bind( this );

    // @private {Object} - The listener that gets added to the pointer when we are pressed
    this._pointerListener = {
      /**
       * Called with 'up' events from the pointer (part of the listener API)
       * @public (scenery-internal)
       *
       * @param {Event} event
       */
      up: function( event ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + self._id + ' pointer up' );
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
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + self._id + ' pointer cancel' );
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
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + self._id + ' pointer move' );
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
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + self._id + ' pointer interrupt' );
        sceneryLog && sceneryLog.InputListener && sceneryLog.push();

        self.interrupt();

        sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
      }
    };

    // @public - the collection of event listeners that can be added to a Node through addAccessibleInputListener,
    // typical usage could look like node.addAccessibleInputListener( pressListener.a11yListener )
    this.a11yListener = {
      click: this.click.bind( this ),
      focus: this.focus.bind( this ),
      blur: this.blur.bind( this )
    };

    // update isOverProperty (not a DerivedProperty because we need to hook to passed-in properties)
    this.overPointers.lengthProperty.link( this.invalidateOver.bind( this ) );

    // update isHoveringProperty (not a DerivedProperty because we need to hook to passed-in properties)
    this.overPointers.lengthProperty.link( this._isHoveringListener );
    this.isPressedProperty.link( this._isHoveringListener );

    // Update isHovering when any pointer's isDownProperty changes.
    // NOTE: overPointers is cleared on dispose, which should remove all of these (interior) listeners)
    this.overPointers.addItemAddedListener( function( pointer ) {
      pointer.isDownProperty.link( self._isHoveringListener );
    } );
    this.overPointers.addItemRemovedListener( function( pointer ) {
      pointer.isDownProperty.unlink( self._isHoveringListener );
    } );

    // update isHighlightedProperty (not a DerivedProperty because we need to hook to passed-in properties)
    this.isHoveringProperty.link( this._isHighlightedListener );
    this.isPressedProperty.link( this._isHighlightedListener );
  }

  scenery.register( 'PressListener', PressListener );

  inherit( PhetioObject, PressListener, {
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
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' down' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.press( event );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Called with 'enter' events (part of the listener API).
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly.
     *
     * @param {Event} event
     */
    enter: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' enter' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.overPointers.push( event.pointer );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Called with 'exit' events (part of the listener API).
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly.
     *
     * @param {Event} event
     */
    exit: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' exit' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // NOTE: We don't require the pointer to be included here, since we may have added the listener after the 'enter'
      // was fired. See https://github.com/phetsims/area-model-common/issues/159 for more details. This may be a
      // no-op, which ObservableArray allows.
      this.overPointers.remove( event.pointer );

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

      // canClick is a subset of canPress, start by checking a11y state that doesn't involve pointers
      if ( !this.canClick() ) {
        return false;
      }

      // Only let presses be started with the correct mouse button.
      if ( event.pointer instanceof Mouse && event.domEvent.button !== this._mouseButton ) {
        return false;
      }

      // We can't attach to a pointer that is already attached.
      if ( this._attach && event.pointer.isAttached() ) {
        return false;
      }

      return true;
    },

    /**
     * Returns whether this PressListener can be clicked from keyboard input.
     * @return {boolean}
     */
    canClick: function() {

      // If this listener is already involved in pressing something, we can't press something
      if ( this.isPressed ) {
        return false;
      }

      // Check whether our options prevent us from starting a press right now.
      if ( !this._canStartPress() ) {
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
     * @param {Node} [targetNode] - If provided, will take the place of the targetNode for this call. Useful for
     *                              forwarded presses.
     * @returns {boolean} success - Returns whether the press was actually started
     */
    press: function( event, targetNode ) {
      assert && assert( event, 'An event is required' );

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' press' );

      if ( !this.canPress( event ) ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' could not press' );
        return false;
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' successful press' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();
      this.phetioStartEvent( 'pressed', {
        x: event.pointer.point.x,
        y: event.pointer.point.y
      } );

      targetNode = targetNode || this._targetNode;

      // Set self properties before the property change, so they are visible to listeners.
      this.pointer = event.pointer;
      this.pressedTrail = targetNode ? targetNode.getUniqueTrail() : event.trail.subtrailTo( event.currentTarget, false );

      this.interrupted = false; // clears the flag (don't set to false before here)

      this.pointer.addInputListener( this._pointerListener, this._attach );
      this._listeningToPointer = true;

      this.pointer.cursor = this._pressCursor;

      this.isPressedProperty.value = true;

      // Notify after everything else is set up
      this._pressListener && this._pressListener( event, this );

      this.phetioEndEvent();
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
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' release' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();
      this.phetioStartEvent( 'released' );

      assert && assert( this.isPressed, 'This listener is not pressed' );

      this.pointer.removeInputListener( this._pointerListener );
      this._listeningToPointer = false;

      this.pointer.cursor = null;

      // Unset self properties after the property change, so they are visible to listeners beforehand.
      this.pointer = null;
      this.pressedTrail = null;

      this.isPressedProperty.value = false;

      // Notify after the rest of release is called in order to prevent it from triggering interrupt().
      this._releaseListener && this._releaseListener( this );

      this.phetioEndEvent();
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
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' drag' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      assert && assert( this.isPressed, 'Can only drag while pressed' );

      this._dragListener && this._dragListener( event, this );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Interrupts the listener, releasing it (canceling behavior).
     * @public
     *
     * This can be called manually, but can also be called through node.interruptSubtreeInput().
     */
    interrupt: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' interrupt' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      if ( this.isPressed ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' interrupting' );
        this.interrupted = true;

        this.release();
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Recomputes the value for isOverProperty. Separate to reduce anonymous function closures.
     * @private
     */
    invalidateOver: function() {
      this.isOverProperty.value = this.overPointers.length > 0;
    },

    /**
     * Recomputes the value for isHoveringProperty. Separate to reduce anonymous function closures.
     * @private
     */
    invalidateHovering: function() {
      var pointers = this.overPointers.getArray();
      for ( var i = 0; i < pointers.length; i++ ) {
        var pointer = pointers[ i ];
        if ( !pointer.isDown || pointer === this.pointer ) {
          this.isHoveringProperty.value = true;
          return;
        }
      }
      this.isHoveringProperty.value = false;
    },

    /**
     * Recomputes the value for isHighlightedProperty. Separate to reduce anonymous function closures.
     * @private
     */
    invalidateHighlighted: function() {
      this.isHighlightedProperty.value = this.isHoveringProperty.value || this.isPressedProperty.value;
    },

    /**
     * Handle the click event from DOM for a11y, see PressListener.a11yListener for more information in this usage.
     * Clicks by setting the over and pressed Properties behind a timeout. When assistive technology is
     * used, the browser may not receive 'down' or 'up' events on buttons - only a single 'click' event. For a11y we
     * need to toggle the pressed state from the single 'click' event.
     *
     * @public - In general not needed to be public, but just used in edge cases to get proper click logic for a11y.
     * @a11y
     */
    click: function( event ) {
      if ( this.canClick() ) {

        // ensure that button is 'over' so listener can be called while button is down
        this.isOverProperty.set( true );
        this.isPressedProperty.set( true );

        var self = this;
        Timer.setTimeout( function() {

          // no longer down, don't reset 'over' so button can be styled as long as it has focus
          self.isPressedProperty.set( false );

          // TODO: how to define the delay for click interval, see https://github.com/phetsims/scenery/issues/831
          // how to handle a11y end listener?
          // endListener && endListener();
        }, this._fireOnHoldInterval );
      }
    },

    /**
     * On focus, button should look 'over'.
     * @private
     * @a11y
     */
    focus: function() {
      this.isOverProperty.value = true;
    },

    /**
     * On blur, the button should no longer look 'over'.
     * @private
     * @a11y
     */
    blur: function() {
      if ( !this.isPressedProperty.get() ) {
        this.isOverProperty.value = false;
      }
    },

    /**
     * Disposes the listener, releasing references. It should not be used after this.
     * @public
     */
    dispose: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' dispose' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // We need to release references to any pointers that are over us.
      this.overPointers.clear();

      if ( this._listeningToPointer ) {
        this.pointer.removeInputListener( this._pointerListener );
      }

      // These Properties could have already been disposed, for example in the sun button hierarchy, see https://github.com/phetsims/sun/issues/372
      if ( !this.isPressedProperty.isDisposed ) {
        this.isPressedProperty.unlink( this._isHighlightedListener );
        this.isPressedProperty.unlink( this._isHoveringListener );
      }
      !this.isHoveringProperty.isDisposed && this.isHoveringProperty.unlink( this._isHighlightedListener );

      PhetioObject.prototype.dispose.call( this );

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }
  } );

  return PressListener;
} );
