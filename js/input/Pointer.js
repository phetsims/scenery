// Copyright 2013-2020, University of Colorado Boulder

/*
 * A pointer is an abstraction that includes a mouse and touch points (and possibly keys). The mouse is a single
 * pointer, and each finger (for touch) is a pointer.
 *
 * Listeners that can be added to the pointer, and events will be fired on these listeners before any listeners are
 * fired on the Node structure. This is typically very useful for tracking dragging behavior (where the pointer may
 * cross areas where the dragged node is not directly below the pointer any more).
 *
 * A valid listener should be an object. If a listener has a property with a Scenery event name (e.g. 'down' or
 * 'touchmove'), then that property will be assumed to be a method and will be called with the Scenery event (like
 * normal input listeners, see Node.addInputListener).
 *
 * Pointers can have one active "attached" listener, which is the main handler for responding to the events. This helps
 * when the main behavior needs to be interrupted, or to determine if the pointer is already in use. Additionally, this
 * can be used to prevent pointers from dragging or interacting with multiple components at the same time.
 *
 * A listener may have an interrupt() method that will attemp to interrupt its behavior. If it is added as an attached
 * listener, then it must have an interrupt() method.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';
import scenery from '../scenery.js';

// constants
// entries when signifying Intent of the pointer, see addIntent
const Intent = Enumeration.byKeys( [
  'DRAG', // listener attached to the pointer will be used for dragging
  'KEYBOARD_DRAG' // listener attached to pointer is for dragging with a keyboard
] );

class Pointer {
  /**
   * @protected (scenery-internal)
   * @abstract
   * @param {Vector2|null} initialPoint
   * @param {boolean} initialDownState
   * @param {string} type - the type of the pointer; can different for each subtype
   */
  constructor( initialPoint, initialDownState, type ) {
    assert && assert( initialPoint === null || initialPoint instanceof Vector2 );
    assert && assert( typeof initialDownState === 'boolean' );

    assert && assert( typeof type === 'string' );

    assert && assert( Object.getPrototypeOf( this ) !== Pointer.prototype, 'Pointer is an abstract class' );

    // @public {Vector2|null} - The location of the pointer in the global coordinate system. If there has no location
    //                          recorded yet, it may be null.
    this.point = initialPoint;

    // @public (read-only) {string}
    // Each Pointer subtype should implement a "type" field that can be checked against for scenery input.
    this.type = type;

    // @public {Trail|null} - The trail that the pointer is currently over (if it has yet been registered). If the
    //                        pointer has not yet registered a trail, it may be null. If the pointer wasn't over any
    //                        specific trail, then a trail with only the display's rootNode will be set.
    this.trail = null;

    // @deprecated @public {BooleanProperty} - Whether this pointer is 'down' (pressed).
    // Will be phased out in https://github.com/phetsims/scenery/issues/803 to something that is specific for the actual
    // mouse/pen button (since this doesn't generalize well to the left/right mouse buttons).
    this.isDownProperty = new BooleanProperty( initialDownState );

    // @public {BooleanProperty} - Whether there is a main listener "attached" to this pointer. This signals that the
    // listener is "doing" something with the pointer, and that it should be interrupted if other actions need to take
    // over the pointer behavior.
    this.attachedProperty = new BooleanProperty( false );

    // @private {Array.<Object>} - All attached listeners (will be activated in order). See top-level documentation for
    //                             information about listener structure.
    this._listeners = [];

    // @private {Object|null} - Our main "attached" listener, if there is one (otherwise null)
    this._attachedListener = null;

    // @private {string|null} - See setCursor() for more information.
    this._cursor = null;

    // @public (scenery-internal) {Event|null} - Recorded and exposed so that it can be provided to events when there
    // is no "immediate" DOM event (e.g. when a node moves UNDER a pointer and triggers a touch-snag).
    this.lastDOMEvent = null;

    // @private {Intent[]]} - A Pointer can be assigned an intent when a listener is attached to initiate or prevent
    // certain behavior for the life of the listener. Other listeners can observe the Intents on the Pointer and
    // react accordingly
    this._intents = [];

    // @private {boolean}
    this._pointerCaptured = false;

    // @private {null|Object} - listeners attached to this pointer that clear the this._intent after input in
    // reserveForDrag functions, referenced so they can be removed on disposal
    this._listenerForDragReserve = null;
    this._listenerForKeyboardDragReserve = null;
  }

  /**
   * Sets a cursor that takes precedence over cursor values specified on the pointer's trail.
   * @public
   *
   * Typically this can be set when a drag starts (and returned to null when the drag ends), so that the cursor won't
   * change while dragging (regardless of what is actually under the pointer). This generally will only apply to the
   * Mouse subtype of Pointer.
   *
   * NOTE: Consider setting this only for attached listeners in the future (or have a cursor field on pointers).
   *
   * @param {string|null} cursor
   * @returns {Pointer} - For chaining
   */
  setCursor( cursor ) {
    this._cursor = cursor;

    return this; // TODO: is chaining actually used? Not that helpful of a pattern with pointers.
  }
  set cursor( value ) { this.setCursor( value ); }

  /**
   * Returns the current cursor override (or null if there is one). See setCursor().
   * @public
   *
   * @returns {string|null}
   */
  getCursor() {
    return this._cursor;
  }
  get cursor() { return this.getCursor(); }

  /**
   * Returns a defensive copy of all listeners attached to this pointer.
   * @public (scenery-internal)
   *
   * @returns {Array.<Object>}
   */
  getListeners() {
    return this._listeners.slice();
  }
  get listeners() { return this.getListeners(); }

  /**
   * Adds an input listener to this pointer. If the attach flag is true, then it will be set as the "attached"
   * listener.
   * @public
   *
   * @param {Object} listener - See top-level documentation for description of the listener API
   * @param {boolean} [attach]
   */
  addInputListener( listener, attach ) {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'addInputListener to ' + this.toString() + ' attach:' + attach );
    sceneryLog && sceneryLog.Pointer && sceneryLog.push();

    assert && assert( listener, 'A listener must be provided' );
    assert && assert( attach === undefined || typeof attach === 'boolean',
      'If provided, the attach parameter should be a boolean value' );

    assert && assert( !_.includes( this._listeners, listener ),
      'Attempted to add an input listener that was already added' );

    this._listeners.push( listener );

    if ( attach ) {
      this.attach( listener );
    }

    sceneryLog && sceneryLog.Pointer && sceneryLog.pop();
  }

  /**
   * Removes an input listener from this pointer.
   * @public
   *
   * @param {Object} listener - See top-level documentation for description of the listener API
   */
  removeInputListener( listener ) {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'removeInputListener to ' + this.toString() );
    sceneryLog && sceneryLog.Pointer && sceneryLog.push();

    assert && assert( listener, 'A listener must be provided' );

    const index = _.indexOf( this._listeners, listener );
    assert && assert( index !== -1, 'Could not find the input listener to remove' );

    // If this listener is our attached listener, also detach it
    if ( this.isAttached() && listener === this._attachedListener ) {
      this.detach( listener );
    }

    this._listeners.splice( index, 1 );

    sceneryLog && sceneryLog.Pointer && sceneryLog.pop();
  }

  /**
   * Returns whether this pointer has an attached (primary) listener.
   * @public
   *
   * @returns {boolean}
   */
  isAttached() {
    return this.attachedProperty.value;
  }

  /**
   * Sets whether this pointer is down/pressed, or up.
   * @public
   *
   * NOTE: Naming convention is for legacy code, would usually have pointer.down
   * TODO: improve name, .setDown( value ) with .down =
   *
   * @param {boolean} value
   */
  set isDown( value ) {
    this.isDownProperty.value = value;
  }

  /**
   * Returns whether this pointer is down/pressed, or up.
   * @public
   *
   * NOTE: Naming convention is for legacy code, would usually have pointer.down
   * TODO: improve name, .isDown() with .down
   *
   * @returns {boolean}
   */
  get isDown() {
    return this.isDownProperty.value;
  }

  /**
   * If there is an attached listener, interrupt it.
   * @public
   *
   * After this executes, this pointer should not be attached.
   */
  interruptAttached() {
    if ( this.isAttached() ) {
      this._attachedListener.interrupt(); // Any listener that uses the 'attach' API should have interrupt()
    }
  }

  /**
   * Interrupts all listeners on this pointer.
   * @public
   */
  interruptAll() {
    const listeners = this._listeners.slice();
    for ( let i = 0; i < listeners.length; i++ ) {
      const listener = listeners[ i ];
      listener.interrupt && listener.interrupt();
    }
  }

  /**
   * Marks the pointer as attached to this listener.
   * @private
   *
   * @param {Object} listener
   */
  attach( listener ) {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Attaching to ' + this.toString() );

    assert && assert( !this.isAttached(), 'Attempted to attach to an already attached pointer' );

    this.attachedProperty.value = true;
    this._attachedListener = listener;
  }

  /**
   * Marks the pointer as detached from a previously attached listener.
   * @private
   *
   * @param {Object} listener
   */
  detach( listener ) {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Detaching from ' + this.toString() );

    assert && assert( this.isAttached(), 'Cannot detach a listener if one is not attached' );
    assert && assert( this._attachedListener === listener, 'Cannot detach a different listener' );

    this.attachedProperty.value = false;
    this._attachedListener = null;
  }

  /**
   * Determines whether the point of the pointer has changed (used in mouse/touch/pen).
   * @protected
   *
   * @param {Vector2} point
   * @returns {boolean}
   */
  hasPointChanged( point ) {
    return this.point !== point && ( !point || !this.point || !this.point.equals( point ) );
  }

  /**
   * Adds an Intent Pointer. By setting Intent, other listeners in the dispatch phase can react accordingly.
   * Note that the Intent can be changed by listeners up the dispatch phase or on the next press. See Intent enum
   * for valid entries.
   * @public
   *
   * @param {Intent} intent
   */
  addIntent( intent ) {
    assert && assert( Intent.includes( intent ), 'trying to set unsupported intent for Pointer' );
    if ( !this._intents.includes( intent ) ) {
      this._intents.push( intent );
    }

    assert && assert( this._intents.length <= Intent.VALUES.length, 'to many Intents saved, memory leak likely' );
  }

  /**
   * Remove an Intent from the Pointer. See addIntent for more information.
   * @public
   *
   * @param {Intent} intent
   */
  removeIntent( intent ) {
    if ( this._intents.includes( intent ) ) {
      const index = this._intents.indexOf( intent );
      this._intents.splice( index, 1 );
    }
  }

  /**
   * Returns whether or not this Pointer has been assigned the provided Intent.
   * @public
   *
   * @param {Intent} intent
   * @returns {boolean}
   */
  hasIntent( intent ) {
    return this._intents.includes( intent );
  }

  /**
   * Set the intent of this Pointer to indicate that it will be used for mouse/touch style dragging, indicating to
   * other listeners in the dispatch phase that behavior may need to change. Adds a listener to the pointer (with
   * self removal) that clears the intent when the pointer receives an "up" event. Should generally be called on
   * the Pointer in response to a down event.
   * @public
   */
  reserveForDrag() {

    // if the Pointer hasn't already been reserved for drag in Input event dispatch, in which
    // case it already has Intent and listener to remove Intent
    if ( !this._intents.includes( Intent.DRAG ) ) {
      this.addIntent( Intent.DRAG );

      const listener = {
        up: event => {
          this.removeIntent( Intent.DRAG );
          this.removeInputListener( this._listenerForDragReserve );
          this._listenerForDragReserve = null;
        }
      };

      assert && assert( this._listenerForDragReserve === null, 'still a listener to reserve pointer, memory leak likely' );
      this._listenerForDragReserve = listener;
      this.addInputListener( this._listenerForDragReserve );
    }
  }

  /**
   * Set the intent of this Pointer to indicate that it will be used for keyboard style dragging, indicating to
   * other listeners in the dispatch that behavior may need to change. Adds a listener to the pointer (with self
   * removal) that clears the intent when the pointer receives a "keyup" or "blur" event. Should generally be called
   * on the Pointer in response to a keydown event.
   * @public
   */
  reserveForKeyboardDrag() {

    if ( !this._intents.includes( Intent.KEYBOARD_DRAG ) ) {
      this.addIntent( Intent.KEYBOARD_DRAG );

      const listener = {
        keyup: event => clearIntent(),

        // clear on blur as well since focus may be lost before we receive a keyup event
        blur: event => clearIntent()
      };

      const clearIntent = () => {
        this.removeIntent( Intent.KEYBOARD_DRAG );
        this.removeInputListener( this._listenerForKeyboardDragReserve );
        this._listenerForKeyboardDragReserve = null;
      };

      assert && assert( this._listenerForDragReserve === null, 'still a listener on Pointer for reserve, memory leak likely' );
      this._listenerForKeyboardDragReserve = listener;
      this.addInputListener( this._listenerForKeyboardDragReserve );
    }
  }

  /**
   * This is called when a capture starts on this pointer. We request it on pointerstart, and if received, we should
   * generally receive events outside the window.
   * @public
   */
  onGotPointerCapture() {
    this._pointerCaptured = true;
  }

  /**
   * This is called when a capture ends on this pointer. This happens normally when the user releases the pointer above
   * the sim or outside, but also in cases where we have NOT received an up/end.
   * @public
   *
   * See https://github.com/phetsims/scenery/issues/1186 for more information. We'll want to interrupt the pointer
   * on this case regardless,
   */
  onLostPointerCapture() {
    if ( this._pointerCaptured ) {
      this.interruptAll();
    }
    this._pointerCaptured = false;
  }

  /**
   * Releases references so it can be garbage collected.
   * @public
   */
  dispose() {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Disposing ' + this.toString() );

    // remove listeners that would clear intent on disposal
    if ( this._listeners.indexOf( this._listenerForDragReserve ) >= 0 ) { this.removeInputListener( this._listenerForDragReserve ); }
    if ( this._listeners.indexOf( this._listenerForKeyboardDragReserve ) >= 0 ) { this.removeInputListener( this._listenerForKeyboardDragReserve ); }

    assert && assert( this._attachedListener === null, 'Attached listeners should be cleared before pointer disposal' );
    assert && assert( this._listeners.length === 0, 'Should not have listeners when a pointer is disposed' );
  }
}

/**
 * @static
 * @public
 */
Pointer.Intent = Intent;

scenery.register( 'Pointer', Pointer );
export default Pointer;