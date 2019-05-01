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

  // modules
  var Action = require( 'AXON/Action' );
  var ActionIO = require( 'AXON/ActionIO' );
  var BooleanProperty = require( 'AXON/BooleanProperty' );
  var DerivedProperty = require( 'AXON/DerivedProperty' );
  var EventIO = require( 'SCENERY/input/EventIO' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Mouse = require( 'SCENERY/input/Mouse' );
  var Node = require( 'SCENERY/nodes/Node' );
  var NullableIO = require( 'TANDEM/types/NullableIO' );
  var ObservableArray = require( 'AXON/ObservableArray' );
  var PhetioObject = require( 'TANDEM/PhetioObject' );
  var scenery = require( 'SCENERY/scenery' );
  var Tandem = require( 'TANDEM/Tandem' );
  var timer = require( 'AXON/timer' );
  var VoidIO = require( 'TANDEM/types/VoidIO' );

  // global
  var globalID = 0;

  // constants - factored out to reduce memory usage, see https://github.com/phetsims/unit-rates/issues/207
  var PressedActionIO = ActionIO( [
    { name: 'event', type: EventIO },
    {
      name: 'targetNode',
      type: VoidIO,
      validator: { isValidValue: function( v ) { return v === null || v instanceof Node; } }
    },
    {
      name: 'callback',
      type: VoidIO,
      validator: { isValidValue: function( v ) { return v === null || typeof v === 'function'; } }
    }
  ] );

  var ReleasedActionIO = ActionIO( [ {
    name: 'event',
    type: NullableIO( EventIO )
  }, {
    name: 'callback',
    type: VoidIO,
    validator: { isValidValue: function( v ) { return v === null || typeof v === 'function'; } }
  } ] );

  // Factor out to reduce memory footprint, see https://github.com/phetsims/tandem/issues/71
  const truePredicate = _.constant( true );

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

      // {function} - Called as press( event: {Event}, listener: {PressListener} ) when this listener's node is pressed
      // (typically from a down event, but can be triggered by other handlers).
      press: _.noop,

      // {function} - Called as release( event: {Event|null}, listener: {PressListener} ) when this listener's node is released
      // Note that an Event arg cannot be guaranteed from this listener. This is, in part, to support interrupt.
      // (pointer up/cancel or interrupt when pressed/after a11y click).
      release: _.noop,

      // {function} - Called as drag( event: {Event}, listener: {PressListener} ) when this listener's node is
      // dragged (move events on the pointer while pressed).
      drag: _.noop,

      // {Node|null} - If provided, the pressedTrail (calculated from the down event) will be replaced with the
      // (sub)trail that ends with the targetNode as the leaf-most Node. This affects the parent coordinate frame
      // computations.
      targetNode: null,

      // {boolean} - If true, this listener will not "press" while the associated pointer is attached, and when pressed,
      // will mark itself as attached to the pointer. If this listener should not be interrupted by others and isn't
      // a "primary" handler of the pointer's behavior, this should be set to false.
      attach: true,

      // {function} - Checks this when trying to start a press. If this function returns false, a press will not be
      // started. Called as canStartPress( event: {Event|null}, listener: {PressListener} ), since sometimes the
      // event may not be available.
      canStartPress: truePredicate,

      // {number} (a11y) - How long something should 'look' pressed after an accessible click input event, in ms
      a11yLooksPressedInterval: 100,

      // {Tandem} - For instrumenting
      tandem: Tandem.required,

      // to support properly passing this to children, see https://github.com/phetsims/tandem/issues/60
      phetioReadOnly: PhetioObject.DEFAULT_OPTIONS.phetioReadOnly,
      phetioFeatured: PhetioObject.DEFAULT_OPTIONS.phetioFeatured
    }, options );

    assert && assert( typeof options.mouseButton === 'number' && options.mouseButton >= 0 && options.mouseButton % 1 === 0,
      'mouseButton should be a non-negative integer' );
    assert && assert( options.pressCursor === null || typeof options.pressCursor === 'string',
      'pressCursor should either be a string or null' );
    assert && assert( typeof options.press === 'function',
      'The press callback should be a function' );
    assert && assert( typeof options.release === 'function',
      'The release callback should be a function' );
    assert && assert( typeof options.drag === 'function',
      'The drag callback should be a function' );
    assert && assert( options.targetNode === null || options.targetNode instanceof Node,
      'If provided, targetNode should be a Node' );
    assert && assert( typeof options.attach === 'boolean', 'attach should be a boolean' );
    assert && assert( typeof options.a11yLooksPressedInterval === 'number',
      'a11yLooksPressedInterval should be a number' );

    // @private {number} - Unique global ID for this listener
    this._id = globalID++;

    sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' construction' );

    // @private {number}
    this._mouseButton = options.mouseButton;
    this._a11yLooksPressedInterval = options.a11yLooksPressedInterval;

    // @private {string|null}
    this._pressCursor = options.pressCursor;

    // @private {function}
    this._pressListener = options.press;
    this._releaseListener = options.release;
    this._dragListener = options.drag;
    this._canStartPress = options.canStartPress;

    // @private {Node|null}
    this._targetNode = options.targetNode;

    // @private {boolean}
    this._attach = options.attach;

    // @public {ObservableArray.<Pointer>} - Contains all pointers that are over our button. Tracked by adding with
    // 'enter' events and removing with 'exit' events.
    this.overPointers = new ObservableArray();

    // @public {Property.<Boolean>} [read-only] - Tracks whether this listener's node is "pressed" or not
    this.isPressedProperty = new BooleanProperty( false, { reentrant: true } );

    // @public {Property.<boolean>} ]read-only] - It will be set to true when at least one pointer is over the listener.
    this.isOverProperty = new BooleanProperty( false );

    // @public {Property.<boolean>} [read-only] - It will be set to true when either:
    //   1. The listener is pressed and the pointer that is pressing is over the listener.
    //   2. There is at least one unpressed pointer that is over the listener.
    this.isHoveringProperty = new BooleanProperty( false );

    // @public {Property.<Boolean>} [read-only] - It will be set to true when either:
    //   1. The listener is pressed.
    //   2. There is at least one unpressed pointer that is over the listener.
    // This is essentially true when ( isPressed || isHovering ).
    this.isHighlightedProperty = new BooleanProperty( false );

    // @public {Property.<boolean>} [read-only] - Whether the listener has focus (should appear to be over)
    this.isFocusedProperty = new BooleanProperty( false );

    // @public {Pointer|null} [read-only] - The current pointer, or null when not pressed.
    this.pointer = null;

    // @public {Trail|null} [read-only] - The Trail for the press, with no descendant nodes past the currentTarget
    // or targetNode (if provided). Will be null when not pressed.
    this.pressedTrail = null;

    // @public {boolean} [read-only] - Whether the last press was interrupted. Will be valid until the next press.
    this.interrupted = false;

    // @private {boolean} - Whether our pointer listener is referenced by the pointer (need to have a flag due to
    //                      handling disposal properly).
    this._listeningToPointer = false;

    // @private {function} - isHoveringProperty updates (not a DerivedProperty because we need to hook to passed-in
    // properties)
    this._isHoveringListener = this.invalidateHovering.bind( this );

    // @private {function} - isHighlightedProperty updates (not a DerivedProperty because we need to hook to passed-in
    // properties)
    this._isHighlightedListener = this.invalidateHighlighted.bind( this );

    // @public {BooleanProperty} [read-only] - Whether or not a press is being processed from an a11y click input event.
    this.a11yClickingProperty = new BooleanProperty( false );

    // @public {BooleanProperty} [read-only] - This Property was added for a11y. It tracks whether or not the button
    // should "look" down. This will be true if downProperty is true or if an a11y click is in progress. For an a11y
    // click, the listeners are fired right away but the button will look down for as long as a11yLooksPressedInterval.
    // See PressListener.click() for more details.
    this.looksPressedProperty = DerivedProperty.or( [ this.a11yClickingProperty, this.isPressedProperty ] );

    // @private {function|null} - When a11y clicking begins, this will be added to a timeout so that the
    // a11yClickingProperty is updated after some delay. This is required since an assistive device (like a switch) may
    // send "click" events directly instead of keydown/keyup pairs. If a click initiates while already in progress,
    // this listener will be removed to start the timeout over. null until timout is added.
    this._a11yClickingTimeoutListener = null;

    // @private {Object} - The listener that gets added to the pointer when we are pressed
    this._pointerListener = {
      up: this.pointerUp.bind( this ),
      cancel: this.pointerCancel.bind( this ),
      move: this.pointerMove.bind( this ),
      interrupt: this.pointerInterrupt.bind( this )
    };

    // @private {Action} - Executed on press event
    // The main implementation of "press" handling is implemented as a callback to the Action, so things are nested
    // nicely for phet-io.
    this._pressedAction = new Action( this.onPress.bind( this ), {
      tandem: options.tandem.createTandem( 'pressedAction' ),
      phetioDocumentation: 'Executes whenever a press occurs. The first argument when executing can be ' +
                           'used to convey info about the Event.',
      phetioReadOnly: options.phetioReadOnly,
      phetioFeatured: options.phetioFeatured,
      phetioEventType: PhetioObject.EventType.USER,
      phetioType: PressedActionIO
    } );

    // @private {Action} - Executed on release event
    // The main implementation of "release" handling is implemented as a callback to the Action, so things are nested
    // nicely for phet-io.
    this._releasedAction = new Action( this.onRelease.bind( this ), {
      tandem: options.tandem.createTandem( 'releasedAction' ),
      phetioDocumentation: 'Executes whenever a release occurs.',
      phetioReadOnly: options.phetioReadOnly,
      phetioFeatured: options.phetioFeatured,
      phetioEventType: PhetioObject.EventType.USER,

      phetioType: ReleasedActionIO
    } );

    // update isOverProperty (not a DerivedProperty because we need to hook to passed-in properties)
    this.overPointers.lengthProperty.link( this.invalidateOver.bind( this ) );
    this.isFocusedProperty.link( this.invalidateOver.bind( this ) );

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
     * Returns whether a press can be started with a particular event.
     * @public
     *
     * @param {Event} event
     * @returns {boolean}
     */
    canPress: function( event ) {
      return !this.isPressed && this._canStartPress( event, this ) &&
             // Only let presses be started with the correct mouse button.
             ( !( event.pointer instanceof Mouse ) || event.domEvent.button === this._mouseButton ) &&
             // We can't attach to a pointer that is already attached.
             ( !this._attach || !event.pointer.isAttached() );
    },

    /**
     * Returns whether this PressListener can be clicked from keyboard input. This copies part of canPress, but
     * we didn't want to use canClick in canPress because canClick could be overridden in subtypes.
     * @public
     *
     * @returns {boolean}
     */
    canClick: function() {
      // If this listener is already involved in pressing something (or our options predicate returns false) we can't
      // press something.
      return !this.isPressed && this._canStartPress( null, this );
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
     * @param {function} [callback] - to be run at the end of the function, but only on success
     * @returns {boolean} success - Returns whether the press was actually started
     */
    press: function( event, targetNode, callback ) {
      assert && assert( event, 'An event is required' );

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' press' );

      if ( !this.canPress( event ) ) {
        sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' could not press' );
        return false;
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' successful press' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();
      this._pressedAction.execute( event, targetNode || null, callback || null ); // cannot pass undefined into execute call

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
     *
     * @param {function} [event] - scenery Event if there was one. We can't gaurantee an event, in part to support interrupting.
     * @param {function} [callback] - called at the end of the release
     */
    release: function( event, callback ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' release' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this._releasedAction.execute( event || null, callback || null ); // cannot pass undefined to execute call

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

      this._dragListener( event, this );

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

      // handle a11y interrupt
      if ( this.a11yClickingProperty.value ) {
        this.interrupted = true;

        if ( timer.hasListener( this._a11yClickingTimeoutListener ) ) {
          timer.clearTimeout( this._a11yClickingTimeoutListener );
          this.a11yClickingProperty.value = false;
        }
      }
      else if ( this.isPressed ) {

        // handle pointer interrupt
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
      this.isOverProperty.value = this.isFocusedProperty.value || this.overPointers.length > 0;
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
     * Internal code executed as the first step of a press.
     * @private
     *
     * @param {Event} event
     * @param {Node} [targetNode] - If provided, will take the place of the targetNode for this call. Useful for
     *                              forwarded presses.
     * @param {function} [callback] - to be run at the end of the function, but only on success
     */
    onPress: function( event, targetNode, callback ) {
      targetNode = targetNode || this._targetNode;

      // Set this properties before the property change, so they are visible to listeners.
      this.pointer = event.pointer;
      this.pressedTrail = targetNode ? targetNode.getUniqueTrail() : event.trail.subtrailTo( event.currentTarget, false );

      this.interrupted = false; // clears the flag (don't set to false before here)

      this.pointer.addInputListener( this._pointerListener, this._attach );
      this._listeningToPointer = true;

      this.pointer.cursor = this._pressCursor;

      this.isPressedProperty.value = true;

      // Notify after everything else is set up
      this._pressListener( event, this );

      callback && callback();
    },

    /**
     * Internal code executed as the first step of a release.
     * @private
     *
     * @param {function} [event] - scenery Event if there was one
     * @param {function} [callback] - called at the end of the release
     */
    onRelease: function( event, callback ) {
      assert && assert( this.isPressed, 'This listener is not pressed' );

      this.pointer.removeInputListener( this._pointerListener );
      this._listeningToPointer = false;

      this.pointer.cursor = null;

      // Unset this properties after the property change, so they are visible to listeners beforehand.
      this.pointer = null;
      this.pressedTrail = null;

      this.isPressedProperty.value = false;

      // Notify after the rest of release is called in order to prevent it from triggering interrupt().
      this._releaseListener( event, this );

      callback && callback();
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
     * Called with 'up' events from the pointer (part of the listener API)
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly.
     *
     * @param {Event} event
     */
    pointerUp: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' pointer up' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // Since our callback can get queued up and THEN interrupted before this happens, we'll check to make sure we are
      // still pressed by the time we get here. If not pressed, then there is nothing to do.
      // See https://github.com/phetsims/capacitor-lab-basics/issues/251
      if ( this.isPressed ) {
        assert && assert( event.pointer === this.pointer );

        this.release( event );
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Called with 'cancel' events from the pointer (part of the listener API)
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly.
     *
     * @param {Event} event
     */
    pointerCancel: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' pointer cancel' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // Since our callback can get queued up and THEN interrupted before this happens, we'll check to make sure we are
      // still pressed by the time we get here. If not pressed, then there is nothing to do.
      // See https://github.com/phetsims/capacitor-lab-basics/issues/251
      if ( this.isPressed ) {
        assert && assert( event.pointer === this.pointer );

        this.interrupt(); // will mark as interrupted and release()
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Called with 'move' events from the pointer (part of the listener API)
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly.
     *
     * @param {Event} event
     */
    pointerMove: function( event ) {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' pointer move' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      // Since our callback can get queued up and THEN interrupted before this happens, we'll check to make sure we are
      // still pressed by the time we get here. If not pressed, then there is nothing to do.
      // See https://github.com/phetsims/capacitor-lab-basics/issues/251
      if ( this.isPressed ) {
        assert && assert( event.pointer === this.pointer );

        this.drag( event );
      }

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Called when the pointer needs to interrupt its current listener (usually so another can be added).
     * @public (scenery-internal)
     *
     * NOTE: Do not call directly.
     */
    pointerInterrupt: function() {
      sceneryLog && sceneryLog.InputListener && sceneryLog.InputListener( 'PressListener#' + this._id + ' pointer interrupt' );
      sceneryLog && sceneryLog.InputListener && sceneryLog.push();

      this.interrupt();

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    },

    /**
     * Click listener, called when this is treated as an accessible input listener.
     * @public - In general not needed to be public, but just used in edge cases to get proper click logic for a11y.
     * @a11y
     *
     * Handle the click event from DOM for a11y. Clicks by setting the over and pressed Properties behind a timeout.
     * When assistive technology is used, the browser may not receive 'down' or 'up' events on buttons - only a single
     * 'click' event. For a11y we need to toggle the pressed state from the single 'click' event.
     *
     * This will fire listeners immediately, but adds a delay for the a11yClickingProperty so that you can make a
     * button look pressed from a single DOM click event. For example usage, see sun/ButtonModel.looksPressedProperty.
     */
    click: function( event ) {
      if ( this.canClick() ) {
        this.interrupted = false; // clears the flag (don't set to false before here)

        this.a11yClickingProperty.value = true;

        // ensure that button is 'over' so listener can be called while button is down
        this.isFocusedProperty.value = true;
        this.isPressedProperty.value = true;

        // fire the optional callback
        this._pressListener( event, this );

        // no longer down, don't reset 'over' so button can be styled as long as it has focus
        this.isPressedProperty.value = false;

        // fire the callback from options
        this._releaseListener( event, this );

        // press or release listeners may have interrupted this click, if that is the case immediately indicate that
        // clicking interaction is over
        if ( this.interrupted ) {
          this.a11yClickingProperty.value = false;
        }
        else {
          // if we are already clicking, remove the previous timeout - this assumes that clearTimeout is a noop if the
          // listener is no longer attached
          timer.clearTimeout( this._a11yClickingTimeoutListener );

          // now add the timeout back to start over, saving so that it can be removed later
          var self = this;
          this._a11yClickingTimeoutListener = timer.setTimeout( function() {
            self.a11yClickingProperty.value = false;
          }, this._a11yLooksPressedInterval );
        }
      }
    },

    /**
     * Focus listener, called when this is treated as an accessible input listener.
     * @public (scenery-internal)
     * @a11y
     */
    focus: function() {
      // On focus, button should look 'over'.
      this.isFocusedProperty.value = true;
    },

    /**
     * Blur listener, called when this is treated as an accessible input listener.
     * @public (scenery-internal)
     * @a11y
     */
    blur: function() {
      // On blur, the button should no longer look 'over'.
      this.isFocusedProperty.value = false;
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

      this.a11yClickingProperty.dispose();
      this.looksPressedProperty.dispose();

      this._pressedAction.dispose();
      this._releasedAction.dispose();

      sceneryLog && sceneryLog.InputListener && sceneryLog.pop();
    }
  } );

  return PressListener;
} );
