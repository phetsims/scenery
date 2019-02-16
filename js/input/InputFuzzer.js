// Copyright 2018, University of Colorado Boulder

/**
 * For generating random mouse/touch input to a Display, to hopefully discover bugs in an automated fashion.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Random = require( 'DOT/Random' );
  var scenery = require( 'SCENERY/scenery' );
  var Vector2 = require( 'DOT/Vector2' );

  /**
   * @constructor
   *
   * @param {Display} display
   * @param {number} seed
   */
  function InputFuzzer( display, seed ) {
    var self = this;

    // @private {Display}
    this.display = display;

    // @private {Array.<Object>} - { id: {number}, position: {Vector2} }
    this.touches = [];

    // @private {number}
    this.nextTouchID = 1;

    // @private {boolean}
    this.isMouseDown = false;

    // @private {Vector2} - Starts at 0,0, because why not
    this.mousePosition = new Vector2( 0, 0 );

    // @private {Random}
    this.random = new Random( { seed: seed } );

    // @private {function} - All of the various actions that may be options at certain times.
    this.mouseToggleAction = function() {
      self.mouseToggle();
    };
    this.mouseMoveAction = function() {
      self.mouseMove();
    };
    this.touchStartAction = function() {
      var touch = self.createTouch( self.getRandomPosition() );
      self.touchStart( touch );
    };
    this.touchMoveAction = function() {
      var touch = self.random.sample( self.touches );
      self.touchMove( touch );
    };
    this.touchEndAction = function() {
      var touch = self.random.sample( self.touches );
      self.touchEnd( touch );
      self.removeTouch( touch );
    };
    this.touchCancelAction = function() {
      var touch = self.random.sample( self.touches );
      self.touchCancel( touch );
      self.removeTouch( touch );
    };
  }

  scenery.register( 'InputFuzzer', InputFuzzer );

  return inherit( Object, InputFuzzer, {
    /**
     * Sends a certain (expected) number of random events through the input system for the display.
     * @public
     *
     * @param {number} averageEventCount
     * @param {boolean} allowMouse
     * @param {boolean} allowTouch
     * @param {number} maximumPointerCount
     */
    fuzzEvents: function( averageEventCount, allowMouse, allowTouch, maximumPointerCount ) {
      assert && assert( averageEventCount > 0, 'averageEventCount must be positive: ' + averageEventCount );

      // run a variable number of events, with a certain chance of bailing out (so no events are possible)
      // models a geometric distribution of events
      // See https://github.com/phetsims/joist/issues/343 for notes on the distribution.
      while ( this.random.nextDouble() < 1 - 1 / ( averageEventCount + 1 ) ) {
        var activePointerCount = this.touches.length + ( this.isMouseDown ? 1 : 0 ); // 1 extra for the mouse if down
        var canAddPointer = activePointerCount < maximumPointerCount;

        var potentialActions = [];

        if ( allowMouse ) {
          // We could always mouse up/move (if we are down), but can't 'down/move' without being able to add a pointer
          if ( this.isMouseDown || canAddPointer ) {
            potentialActions.push( this.mouseToggleAction );
            potentialActions.push( this.mouseMoveAction );
          }
        }

        if ( allowTouch ) {
          if ( canAddPointer ) {
            potentialActions.push( this.touchStartAction );
          }
          if ( this.touches.length ) {
            potentialActions.push( this.random.nextDouble() < 0.8 ? this.touchEndAction : this.touchCancelAction );
            potentialActions.push( this.touchMoveAction );
          }
        }

        var action = this.random.sample( potentialActions );
        action();
      }
    },

    /**
     * Creates a touch event from multiple touch "items".
     * @private
     *
     * @param {string} type - The main event type, e.g. 'touchmove'.
     * @param {Array.<Object>} touches - A subset of touch objects stored on the fuzzer itself.
     * @returns {Event} - If possible a TouchEvent, but may be a CustomEvent
     */
    createTouchEvent: function( type, touches ) {
      var domElement = this.display.domElement;

      // A specification that looks like a Touch object (and may be used to create one)
      var touchItems = touches.map( function( touch ) {
        return {
          identifier: touch.id,
          target: domElement,
          clientX: touch.position.x,
          clientY: touch.position.y
        };
      } );

      // Check if we can use Touch/TouchEvent constructors, see https://www.chromestatus.com/feature/4923255479599104
      if ( window.Touch !== undefined &&
           window.TouchEvent !== undefined &&
           window.Touch.length === 1 &&
           window.TouchEvent.length === 1 ) {
        var rawTouches = touchItems.map( function( touchItem ) {
          return new window.Touch( touchItem );
        } );

        return new window.TouchEvent( type, {
          cancelable: true,
          bubbles: true,
          touches: rawTouches,
          targetTouches: [],
          changedTouches: rawTouches,
          shiftKey: false // TODO: Do we need this?
        } );
      }
      // Otherwise, use a CustomEvent and "fake" it.
      else {
        var event = document.createEvent( 'CustomEvent' );
        event.initCustomEvent( type, true, true, {
          touches: touchItems,
          targetTouches: [],
          changedTouches: touchItems
        } );
        return event;
      }
    },

    /**
     * Returns a random position somewhere in the display's global coordinates.
     * @private
     *
     * @returns {Vector2}
     */
    getRandomPosition: function() {
      return new Vector2(
        Math.floor( this.random.nextDouble() * this.display.width ),
        Math.floor( this.random.nextDouble() * this.display.height )
      );
    },

    /**
     * Creates a touch from a position (and adds it).
     * @private
     *
     * @param {Vector2} position
     * @returns {Object}
     */
    createTouch: function( position ) {
      var touch = {
        id: this.nextTouchID++,
        position: position
      };
      this.touches.push( touch );
      return touch;
    },

    /**
     * Removes a touch from our list.
     * @private
     *
     * @param {Object} touch
     */
    removeTouch: function( touch ) {
      this.touches.splice( this.touches.indexOf( touch ), 1 );
    },

    /**
     * Triggers a touchStart for the given touch.
     * @private
     *
     * @param {Object} touch
     */
    touchStart: function( touch ) {
      var event = this.createTouchEvent( 'touchstart', [ touch ] );

      this.display._input.validatePointers();
      this.display._input.touchStart( touch.id, touch.position, event );
    },

    /**
     * Triggers a touchMove for the given touch (to a random position in the display).
     * @private
     *
     * @param {Object} touch
     */
    touchMove: function( touch ) {
      touch.position = this.getRandomPosition();

      var event = this.createTouchEvent( 'touchmove', [ touch ] );

      this.display._input.validatePointers();
      this.display._input.touchMove( touch.id, touch.position, event );
    },

    /**
     * Triggers a touchEnd for the given touch.
     * @private
     *
     * @param {Object} touch
     */
    touchEnd: function( touch ) {
      var event = this.createTouchEvent( 'touchend', [ touch ] );

      this.display._input.validatePointers();
      this.display._input.touchEnd( touch.id, touch.position, event );
    },

    /**
     * Triggers a touchCancel for the given touch.
     * @private
     *
     * @param {Object} touch
     */
    touchCancel: function( touch ) {
      var event = this.createTouchEvent( 'touchcancel', [ touch ] );

      this.display._input.validatePointers();
      this.display._input.touchCancel( touch.id, touch.position, event );
    },

    /**
     * Triggers a mouse toggle (switching from down => up or vice versa).
     * @private
     */
    mouseToggle: function() {
      var domEvent = document.createEvent( 'MouseEvent' );

      // technically deprecated, but DOM4 event constructors not out yet. people on #whatwg said to use it
      domEvent.initMouseEvent( this.isMouseDown ? 'mouseup' : 'mousedown', true, true, window, 1, // click count
        this.mousePosition.x, this.mousePosition.y, this.mousePosition.x, this.mousePosition.y,
        false, false, false, false,
        0, // button
        null );

      this.display._input.validatePointers();

      if ( this.isMouseDown ) {
        this.display._input.mouseUp( this.mousePosition, domEvent );
        this.isMouseDown = false;
      }
      else {
        this.display._input.mouseDown( this.mousePosition, domEvent );
        this.isMouseDown = true;
      }
    },

    /**
     * Triggers a mouse move (to a random position in the display).
     * @private
     */
    mouseMove: function() {
      this.mousePosition = this.getRandomPosition();

      // our move event
      var domEvent = document.createEvent( 'MouseEvent' ); // not 'MouseEvents' according to DOM Level 3 spec

      // technically deprecated, but DOM4 event constructors not out yet. people on #whatwg said to use it
      domEvent.initMouseEvent( 'mousemove', true, true, window, 0, // click count
        this.mousePosition.x, this.mousePosition.y, this.mousePosition.x, this.mousePosition.y,
        false, false, false, false,
        0, // button
        null );

      this.display._input.validatePointers();
      this.display._input.mouseMove( this.mousePosition, domEvent );
    }
  } );
} );
