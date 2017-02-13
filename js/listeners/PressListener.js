// Copyright 2013-2016, University of Colorado Boulder

/**
 * TODO: doc
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
   * TODO: doc
   */
  function PressListener( options ) {
    var self = this;

    options = _.extend( {
      mouseButton: 0, // {number} - Restricts to the specific mouse button (but allows any touch)
      pressCursor: 'pointer', // {string} - Sets the pointer cursor to this value when dragging
      press: null,
      release: null,
      drag: null,
      isPressedProperty: null // If set, allow usage instead of creating a new property
    }, options );

    assert && assert( !options.isPressedProperty || options.isPressedProperty.value === false );

    // @public {Property.<Boolean>} [read-only] - Whether this listener is currently in the 'pressed' state or not
    this.isPressedProperty = options.isPressedProperty || new Property( false );

    // @public {Pointer|null} [read-only] - The current pointer (if pressed)
    this.pointer = null;

    // @public {Trail|null} [read-only] - The Trail for the press, possibly including descendant nodes past the currentTarget
    this.fullPressedTrail = null;

    // @public {Trail|null} [read-only] - The Trail for the press, with no descendant nodes past the currentTarget
    this.pressedTrail = null;

    // @public {boolean} [read-only] - Whether the last press was interrupted. Will be valid until the next press
    this.wasInterrupted = false;

    // @private {number} [read-only]
    this._mouseButton = options.mouseButton;

    // @private {string} [read-only]
    this._pressCursor = options.pressCursor;

    // @private
    this._pressListener = options.press;
    this._releaseListener = options.release;
    this._dragListener = options.drag;

    // @private {Object}
    // TODO: hmmm, how to do this?
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
      this.pressedTrail = event.trail.subtrailTo( event.currentTarget, false );
      this.wasInterrupted = false;

      this.isPressedProperty.value = true;

      this.pointer.addInputListener( this._pointerListener );
      this.pointer.cursor = this._pressCursor;

      this._pressListener && this._pressListener(); // TODO: args?
    },

    // TODO: can we do this without the event? Maybe pointer, trail, etc?
    release: function() {
      assert && assert( this.isPressed, 'This listener is not pressed' );

      this.isPressedProperty.value = false;

      this.pointer.removeInputListener( this._pointerListener );
      this.pointer.cursor = null;

      // Unset self properties after the property change, so they are visible to listeners beforehand.
      this.pointer = null;
      this.fullPressedTrail = null;
      this.pressedTrail = null;

      this._releaseListener && this._releaseListener();
    },

    drag: function( event ) {
      assert && assert( this.isPressed, 'Can only drag while pressed' );

      this._dragListener && this._dragListener(); // TODO: args
    },

    interrupt: function() {
      if ( this.isPressed ) {
        this.wasInterrupted = true;

        this.release();
      }
    }
  } );

  return PressListener;
} );
