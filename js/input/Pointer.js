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
import inherit from '../../../phet-core/js/inherit.js';
import scenery from '../scenery.js';

// constants
// entries when signifying Intent of the pointer, see setIntent
const Intent = Enumeration.byKeys( [
  'DRAG', // listener attached to the pointer will be used for dragging
  'MULTI_DRAG', // listener attached to pointer and more than one item can be dragged at at a time with multitouch
  'KEYBOARD_DRAG' // listener attached to pointer is for dragging with a keyboard
] );

/**
 * @constructor
 * @protected (scenery-internal)
 * @abstract
 * @param {Vector2|null} initialPoint
 * @param {boolean} initialDownState
 * @param {string} type - the type of the pointer; can different for each subtype
 */
function Pointer( initialPoint, initialDownState, type ) {
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

  // @private {null|Intent} - A Pointer can be assigned an intent when a listener is attached to initiate or prevent
  // certain behavior for the life of the listener. Other listeners can observe the Intent on the Pointer and
  // react accordingly
  this._intent = null;

  // @private {null|Object} - listeners attached to this pointer that clear the this._intent after input in
  // reserveForDrag functions, referenced so they can be removed on disposal
  this._listenerForDragReserve = null;
  this._listenerForKeyboardDragReserve = null;
}

scenery.register( 'Pointer', Pointer );

inherit( Object, Pointer, {

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
  setCursor: function( cursor ) {
    this._cursor = cursor;

    return this; // TODO: is chaining actually used? Not that helpful of a pattern with pointers.
  },
  set cursor( value ) { return this.setCursor( value ); },

  /**
   * Returns the current cursor override (or null if there is one). See setCursor().
   * @public
   *
   * @returns {string|null}
   */
  getCursor: function() {
    return this._cursor;
  },
  get cursor() { return this.getCursor(); },

  /**
   * Returns a defensive copy of all listeners attached to this pointer.
   * @public (scenery-internal)
   *
   * @returns {Array.<Object>}
   */
  getListeners: function() {
    return this._listeners.slice();
  },
  get listeners() { return this.getListeners(); },

  /**
   * Adds an input listener to this pointer. If the attach flag is true, then it will be set as the "attached"
   * listener.
   * @public
   *
   * @param {Object} listener - See top-level documentation for description of the listener API
   * @param {boolean} [attach]
   */
  addInputListener: function( listener, attach ) {
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
  },

  /**
   * Removes an input listener from this pointer.
   * @public
   *
   * @param {Object} listener - See top-level documentation for description of the listener API
   */
  removeInputListener: function( listener ) {
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
  },

  /**
   * Returns whether this pointer has an attached (primary) listener.
   * @public
   *
   * @returns {boolean}
   */
  isAttached: function() {
    return this.attachedProperty.value;
  },

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
  },

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
  },

  /**
   * If there is an attached listener, interrupt it.
   * @public
   *
   * After this executes, this pointer should not be attached.
   */
  interruptAttached: function() {
    if ( this.isAttached() ) {
      this._attachedListener.interrupt(); // Any listener that uses the 'attach' API should have interrupt()
    }
  },

  /**
   * Interrupts all listeners on this pointer.
   * @public
   */
  interruptAll: function() {
    const listeners = this._listeners.slice();
    for ( let i = 0; i < listeners.length; i++ ) {
      const listener = listeners[ i ];
      listener.interrupt && listener.interrupt();
    }
  },

  /**
   * Marks the pointer as attached to this listener.
   * @private
   *
   * @param {Object} listener
   */
  attach: function( listener ) {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Attaching to ' + this.toString() );

    assert && assert( !this.isAttached(), 'Attempted to attach to an already attached pointer' );

    this.attachedProperty.value = true;
    this._attachedListener = listener;
  },

  /**
   * Marks the pointer as detached from a previously attached listener.
   * @private
   *
   * @param {Object} listener
   */
  detach: function( listener ) {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Detaching from ' + this.toString() );

    assert && assert( this.isAttached(), 'Cannot detach a listener if one is not attached' );
    assert && assert( this._attachedListener === listener, 'Cannot detach a different listener' );

    this.attachedProperty.value = false;
    this._attachedListener = null;
  },

  /**
   * Determines whether the point of the pointer has changed (used in mouse/touch/pen).
   * @protected
   *
   * @param {Vector2} point
   * @returns {boolean}
   */
  hasPointChanged: function( point ) {
    return this.point !== point && ( !point || !this.point || !this.point.equals( point ) );
  },

  /**
   * Sets the Intent on the Pointer. By setting Intent, other listeners in the dispatch phase can react accordingly.
   * Note that the Intent can be changed by listeners up the dispatch phase or on the next press. See Intent enum
   * for valid entries.
   * @param {Intent|null} intent
   */
  setIntent: function( intent ) {
    assert && assert( intent === null || Intent.includes( intent ), 'trying to set unsupported intent for Pointer' );
    this._intent = intent;
  },

  getIntent: function() {
    return this._intent;
  },
  get intent() { return this.getIntent(); },

  /**
   * Set the intent of this Pointer to indicate that it will be used for mouse/touch style dragging, indicating to
   * other listeners in the dispatch phase that behavior may need to change. Adds a listener to the pointer (with
   * self removal) that clears the intent when the pointer receives an "up" event. Should generally be called on
   * the Pointer in response to a down event.
   * @public
   */
  reserveForDrag: function() {
    this.setIntent( Intent.DRAG );

    this._listenerForDragReserve = {
      up: event => {
        this.setIntent( null );
        this.removeInputListener( this._listenerForDragReserve );
      }
    };
    this.addInputListener( this._listenerForDragReserve );
  },

  /**
   * Set the intent of this Pointer to indicate that it will be used for keyboard style dragging, indicating to
   * other listeners in the dispatch that behavior may need to change. Adds a listener to the pointer (with self
   * removal) that clears the intent when the pointer receives a "keyup" or "blur" event. Should generally be called
   * on the Pointer in response to a keydown event.
   * @public
   */
  reserveForKeyboardDrag: function() {
    this.setIntent( Intent.KEYBOARD_DRAG );

    const clearIntent = () => {
      this.setIntent( null );
      this.removeInputListener( this._listenerForKeyboardDragReserve );
    };

    // clear on blur as well since focus may be lost before we receive a keyup event
    this._listenerForKeyboardDragReserve = {
      keyup: event => clearIntent(),
      blur: event => clearIntent()
    };
    this.addInputListener( this._listenerForKeyboardDragReserve );
  },

  /**
   * Releases references so it can be garbage collected.
   * @public
   */
  dispose: function() {
    sceneryLog && sceneryLog.Pointer && sceneryLog.Pointer( 'Disposing ' + this.toString() );

    // remove listeners that would clear intent on disposal
    if ( this._listeners.indexOf( this._listenerForDragReserve ) >= 0 ) { this.removeInputListener( this._listenerForDragReserve ); }
    if ( this._listeners.indexOf( this._listenerForKeyboardDragReserve ) >= 0 ) { this.removeInputListener( this._listenerForKeyboardDragReserve ); }

    assert && assert( this._attachedListener === null, 'Attached listeners should be cleared before pointer disposal' );
    assert && assert( this._listeners.length === 0, 'Should not have listeners when a pointer is disposed' );
  }
}, {

  /**
   * @static
   * @public
   */
  Intent: Intent
} );

export default Pointer;