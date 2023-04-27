// Copyright 2018-2023, University of Colorado Boulder

/**
 * For generating random mouse/touch input to a Display, to hopefully discover bugs in an automated fashion.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Random from '../../../dot/js/Random.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { EventContext, scenery } from '../imports.js';

class InputFuzzer {
  /**
   * @param {Display} display
   * @param {number} seed
   */
  constructor( display, seed ) {

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
    this.mouseToggleAction = () => {
      this.mouseToggle();
    };
    this.mouseMoveAction = () => {
      this.mouseMove();
    };
    this.touchStartAction = () => {
      const touch = this.createTouch( this.getRandomPosition() );
      this.touchStart( touch );
    };
    this.touchMoveAction = () => {
      const touch = this.random.sample( this.touches );
      this.touchMove( touch );
    };
    this.touchEndAction = () => {
      const touch = this.random.sample( this.touches );
      this.touchEnd( touch );
      this.removeTouch( touch );
    };
    this.touchCancelAction = () => {
      const touch = this.random.sample( this.touches );
      this.touchCancel( touch );
      this.removeTouch( touch );
    };
  }

  /**
   * Sends a certain (expected) number of random events through the input system for the display.
   * @public
   *
   * @param {number} averageEventCount
   * @param {boolean} allowMouse
   * @param {boolean} allowTouch
   * @param {number} maximumPointerCount
   */
  fuzzEvents( averageEventCount, allowMouse, allowTouch, maximumPointerCount ) {
    assert && assert( averageEventCount > 0, `averageEventCount must be positive: ${averageEventCount}` );

    this.display._input.currentlyFiringEvents = true;

    // run a variable number of events, with a certain chance of bailing out (so no events are possible)
    // models a geometric distribution of events
    // See https://github.com/phetsims/joist/issues/343 for notes on the distribution.
    while ( this.random.nextDouble() < 1 - 1 / ( averageEventCount + 1 ) ) {
      const activePointerCount = this.touches.length + ( this.isMouseDown ? 1 : 0 ); // 1 extra for the mouse if down
      const canAddPointer = activePointerCount < maximumPointerCount;

      const potentialActions = [];

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

      const action = this.random.sample( potentialActions );
      action();
    }

    this.display._input.currentlyFiringEvents = false;

    // Since we do a lock-out to stop reentrant events above, we'll need to fire any batched events that have accumulated
    // see https://github.com/phetsims/scenery/issues/1497. We'll likely get some focus events that need to fire
    // for correctness, before we continue on.
    this.display._input.fireBatchedEvents();
  }

  /**
   * Creates a touch event from multiple touch "items".
   * @private
   *
   * @param {string} type - The main event type, e.g. 'touchmove'.
   * @param {Array.<Object>} touches - A subset of touch objects stored on the fuzzer itself.
   * @returns {Event} - If possible a TouchEvent, but may be a CustomEvent
   */
  createTouchEvent( type, touches ) {
    const domElement = this.display.domElement;

    // A specification that looks like a Touch object (and may be used to create one)
    const touchItems = touches.map( touch => ( {
      identifier: touch.id,
      target: domElement,
      clientX: touch.position.x,
      clientY: touch.position.y
    } ) );

    // Check if we can use Touch/TouchEvent constructors, see https://www.chromestatus.com/feature/4923255479599104
    if ( window.Touch !== undefined &&
         window.TouchEvent !== undefined &&
         window.Touch.length === 1 &&
         window.TouchEvent.length === 1 ) {
      const rawTouches = touchItems.map( touchItem => new window.Touch( touchItem ) );

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
      const event = document.createEvent( 'CustomEvent' );
      event.initCustomEvent( type, true, true, {
        touches: touchItems,
        targetTouches: [],
        changedTouches: touchItems
      } );
      return event;
    }
  }

  /**
   * Returns a random position somewhere in the display's global coordinates.
   * @private
   *
   * @returns {Vector2}
   */
  getRandomPosition() {
    return new Vector2(
      Math.floor( this.random.nextDouble() * this.display.width ),
      Math.floor( this.random.nextDouble() * this.display.height )
    );
  }

  /**
   * Creates a touch from a position (and adds it).
   * @private
   *
   * @param {Vector2} position
   * @returns {Object}
   */
  createTouch( position ) {
    const touch = {
      id: this.nextTouchID++,
      position: position
    };
    this.touches.push( touch );
    return touch;
  }

  /**
   * Removes a touch from our list.
   * @private
   *
   * @param {Object} touch
   */
  removeTouch( touch ) {
    this.touches.splice( this.touches.indexOf( touch ), 1 );
  }

  /**
   * Triggers a touchStart for the given touch.
   * @private
   *
   * @param {Object} touch
   */
  touchStart( touch ) {
    const event = this.createTouchEvent( 'touchstart', [ touch ] );

    this.display._input.validatePointers();
    this.display._input.touchStart( touch.id, touch.position, new EventContext( event ) );
  }

  /**
   * Triggers a touchMove for the given touch (to a random position in the display).
   * @private
   *
   * @param {Object} touch
   */
  touchMove( touch ) {
    touch.position = this.getRandomPosition();

    const event = this.createTouchEvent( 'touchmove', [ touch ] );

    this.display._input.validatePointers();
    this.display._input.touchMove( touch.id, touch.position, new EventContext( event ) );
  }

  /**
   * Triggers a touchEnd for the given touch.
   * @private
   *
   * @param {Object} touch
   */
  touchEnd( touch ) {
    const event = this.createTouchEvent( 'touchend', [ touch ] );

    this.display._input.validatePointers();
    this.display._input.touchEnd( touch.id, touch.position, new EventContext( event ) );
  }

  /**
   * Triggers a touchCancel for the given touch.
   * @private
   *
   * @param {Object} touch
   */
  touchCancel( touch ) {
    const event = this.createTouchEvent( 'touchcancel', [ touch ] );

    this.display._input.validatePointers();
    this.display._input.touchCancel( touch.id, touch.position, new EventContext( event ) );
  }

  /**
   * Triggers a mouse toggle (switching from down => up or vice versa).
   * @private
   */
  mouseToggle() {
    const domEvent = document.createEvent( 'MouseEvent' );

    // technically deprecated, but DOM4 event constructors not out yet. people on #whatwg said to use it
    domEvent.initMouseEvent( this.isMouseDown ? 'mouseup' : 'mousedown', true, true, window, 1, // click count
      this.mousePosition.x, this.mousePosition.y, this.mousePosition.x, this.mousePosition.y,
      false, false, false, false,
      0, // button
      null );

    this.display._input.validatePointers();

    if ( this.isMouseDown ) {
      this.display._input.mouseUp( this.mousePosition, new EventContext( domEvent ) );
      this.isMouseDown = false;
    }
    else {
      this.display._input.mouseDown( null, this.mousePosition, new EventContext( domEvent ) );
      this.isMouseDown = true;
    }
  }

  /**
   * Triggers a mouse move (to a random position in the display).
   * @private
   */
  mouseMove() {
    this.mousePosition = this.getRandomPosition();

    // our move event
    const domEvent = document.createEvent( 'MouseEvent' ); // not 'MouseEvents' according to DOM Level 3 spec

    // technically deprecated, but DOM4 event constructors not out yet. people on #whatwg said to use it
    domEvent.initMouseEvent( 'mousemove', true, true, window, 0, // click count
      this.mousePosition.x, this.mousePosition.y, this.mousePosition.x, this.mousePosition.y,
      false, false, false, false,
      0, // button
      null );

    this.display._input.validatePointers();
    this.display._input.mouseMove( this.mousePosition, new EventContext( domEvent ) );
  }
}

scenery.register( 'InputFuzzer', InputFuzzer );
export default InputFuzzer;