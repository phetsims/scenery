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

      // {Function|null} - Called as drag( event: {scenery.Event} ) when this listener's node is dragged (move events
      // on the pointer while pressed).
      drag: null,

      // {Property.<Boolean>} - If provided, this property will be used to track whether this listener's node is
      // "pressed" or not.
      isPressedProperty: new Property( false ),

      // TODO doc
      targetNode: null
    }, options );

    assert && assert( options.isPressedProperty.value === false,
      'If a custom isPressedProperty is provided, it must be false initially' );

    // @public {Property.<Boolean>} [read-only] - Whether this listener is currently in the 'pressed' state or not
    this.isPressedProperty = options.isPressedProperty;

    // @public {Pointer|null} [read-only] - The current pointer (if pressed)
    this.pointer = null;

    // @public {Trail|null} [read-only] - The Trail for the press, possibly including descendant nodes past the currentTarget
    // TODO: evaluate how often this would be used (event.trail may be sufficient in most cases)
    this.fullPressedTrail = null;

    // @public {Trail|null} [read-only] - The Trail for the press, with no descendant nodes past the currentTarget
    this.pressedTrail = null;

    // TODO: doc
    this._targetNode = options.targetNode;

    // @public {boolean} [read-only] - Whether the last press was interrupted. Will be valid until the next press.
    this.wasInterrupted = false;

    // @private (stored options)
    this._mouseButton = options.mouseButton;
    this._pressCursor = options.pressCursor;
    this._pressListener = options.press;
    this._releaseListener = options.release;
    this._dragListener = options.drag;

    this._pointerListenerAttached = false;
    // @private {Object} - The listener that gets added to the pointer when we are pressed
    this._pointerListener = {
      up: function( event ) {
        assert && assert( event.pointer === self.pointer );

        self.release();

        // TODO: should we fill in currentTarget and replace after?
      },

      cancel: function( event ) {
        assert && assert( event.pointer === self.pointer );

        self.wasInterrupted = true;

        self.release();

        // TODO: should we fill in currentTarget and replace after?
      },

      move: function( event ) {
        assert && assert( event.pointer === self.pointer );

        self.drag( event );
        // TODO: should we fill in currentTarget and replace after?
      }
    };
  }

  scenery.register( 'PressListener', PressListener );

  inherit( Object, PressListener, {
    // TODO: doc
    get isPressed() {
      return this.isPressedProperty.value;
    },

    // TODO: doc
    // TODO: evaluate if this should be here?
    get currentTarget() {
      assert && assert( this.isPressListener, 'We have no currentTarget if we are not pressed' );

      return this.trail.lastNode();
    },

    down: function( event ) {
      this.tryPress( event );
    },

    tryPress: function( event ) {
      if ( this.isPressed ) { return; }

      if ( event.pointer.isMouse && event.domEvent.button !== this._mouseButton ) { return; }

      this.press( event );
    },

    // TODO: can we do this without the event? Maybe pointer, trail, etc?
    press: function( event ) {
      assert && assert( !this.isPressed, 'This listener is already pressed' );

      // Set self properties before the property change, so they are visible to listeners.
      this.pointer = event.pointer;
      this.fullPressedTrail = event.trail;
      this.pressedTrail = this._targetNode ? this._targetNode.getUniqueTrail() :
                                             event.trail.subtrailTo( event.currentTarget, false );
      this.wasInterrupted = false;

      this.isPressedProperty.value = true;

      this.pointer.addInputListener( this._pointerListener );
      this._pointerListenerAttached = true;

      this.pointer.cursor = this._pressCursor;

      this._pressListener && this._pressListener( event );
    },

    // TODO: can we do this without the event? Maybe pointer, trail, etc?
    release: function() {
      assert && assert( this.isPressed, 'This listener is not pressed' );

      this.isPressedProperty.value = false;

      this.pointer.removeInputListener( this._pointerListener );
      this._pointerListenerAttached = false;

      this.pointer.cursor = null;

      // Unset self properties after the property change, so they are visible to listeners beforehand.
      this.pointer = null;
      this.fullPressedTrail = null;
      this.pressedTrail = null;

      this._releaseListener && this._releaseListener();
    },

    drag: function( event ) {
      assert && assert( this.isPressed, 'Can only drag while pressed' );

      this._dragListener && this._dragListener( event );
    },

    interrupt: function() {
      if ( this.isPressed ) {
        this.wasInterrupted = true;

        this.release();
      }
    },

    dispose: function() {
      if ( this._pointerListenerAttached ) {
        this.pointer.removeInputListener( this._pointerListener );
      }
    }
  } );

  return PressListener;
} );
