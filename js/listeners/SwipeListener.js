// Copyright 2020, University of Colorado Boulder

/**
 * A prototype listener for accessibility purposes. Intended to be added to the display
 * with the following behavior when the user interacts anywhere on the screen, unless
 * the pointer is already attached.
 *
 * - Swipe right, focus next
 * - Swipe left, focus previous
 * - Double tap, activate focusable item (sending click event)
 * - Press and hold, initiate drag of focused item (forwarding press to item)
 *
 * We hope that the above input strategies will allow BVI users to interact with the sim
 * without the use of a screen reader, but in combination with the self-voicing feature set.
 *
 * PROTOTYPE. DO NOT USE IN PRODUCTION CODE.
 *
 * @author Jesse Greenberg
 */

import Display from '../display/Display.js';
import Pointer from '../input/Pointer.js';
import scenery from '../scenery.js';
import PDOMUtils from '../accessibility/pdom/PDOMUtils.js';
import stepTimer from '../../../axon/js/stepTimer.js';

// constants
const PRESS_AND_HOLD_INTERVAL = 0.75; // in seconds, amount of time to initiate a press and hold gesture
const DOUBLE_TAP_INTERVAL = 0.6; // in seconds, max time between down events that would indicate a click gesture

class SwipeListener {
  constructor() {

    // @private - reference to the pointer taken on down, to watch for the user gesture
    this._pointer = null;

    // @private the position (in global coordinate frame) of the point on initial down
    this.downPoint = null;

    // @private {Vector2} - point of the last Pointer on down
    this.lastPoint = null;
    this.currentPoint = null;
    this.velocity = null;
    this.swipeDistance = null;

    this.firstUp = false;
    this.timeSinceLastDown = 0;

    // @private - list of all pointers that are currently down for this listener - if there are more than one
    // we will allow responding to zoom guestures, but if there is only one pointer we will prevent pan
    // gestures because we are taking over for swipe gestures instead
    this.downPointers = [];

    // amount of time in seconds that a finger has been down on the screen - when this
    // time becomes larger than the interval we forward a drag listener to the
    // display target
    this.holdingTime = 0;

    // @private - a reference to the focused Node so that we can call swipe functions
    // implemented on the Node when a swipe to drag gesture has been initiated
    this.focusedNode = null;

    // @private - listener added to the pointer with atachment to call swipe functions
    // on a particular node with focus
    this._attachedPointerListener = {
      up: event => {
        this.focusedNode && this.focusedNode.swipeEnd && this.focusedNode.swipeEnd.bind( this.focusedNode )( event );

        // remove this listener, call the focusedNode's swipeEnd function
        this.focusedNode = null;
        this._pointer.removeInputListener( this._attachedPointerListener );
        this._pointer = null;
      },

      move: event => {

        // call the focusedNode's swipeDrag function
        this.focusedNode && this.focusedNode.swipeMove && this.focusedNode.swipeMove.bind( this.focusedNode )( event );
      },

      interrupt: event => {
        this.focusedNode = null;
        this._pointer.removeInputListener( this._attachedPointerListener );
        this._pointer = null;
      }
    };

    // @private - added to Pointer on down without attaching so that if the event does result
    // in attachment elsewhere, this listener can be interrupted
    this._pointerListener = {
      up: event => {

        // on all releases, clear references and timers
        this.endSwipe();
        this._pointer = null;

        this.swipeDistance = event.pointer.point.minus( this.downPoint );

        const verticalDistance = this.swipeDistance.y;
        const horizontalDistance = this.swipeDistance.x;
        if ( Math.abs( horizontalDistance ) > 100 && Math.abs( verticalDistance ) < 100 ) {

          // some sort of horizontal swipe
          if ( horizontalDistance > 0 ) {
            PDOMUtils.getNextFocusable( document.body ).focus();
          }
          else {
            PDOMUtils.getPreviousFocusable( document.body ).focus();
          }
        }
        else {

          // potentially a double tap
          if ( this.firstUp ) {
            if ( this.timeSinceLastDown < DOUBLE_TAP_INTERVAL ) {
              this.firstUp = false;
              this.timeSinceLastDown = 0;

              // send a click event to the active element
              const pdomRoot = document.getElementsByClassName( 'pdom-root' )[ 0 ];
              if ( pdomRoot && pdomRoot.contains( document.activeElement ) ) {
                document.activeElement.click();
              }
            }
          }
          else {
            this.firstUp = true;
          }
        }
      },

      move: event => {
        if ( event.pointer.isAttached() ) {
          this.interrupt();
        }
        this.lastPoint = this.currentPoint;
        this.currentPoint = event.pointer.point;
      }
    };

    stepTimer.addListener( this.step.bind( this ) );
  }

  /**
   * Part of the scenery input API.
   *
   * @public (scenery-internal)
   * @param event
   */
  down( event ) {
    event.pointer.addIntent( Pointer.Intent.DRAG );
    this.downPointers.push( event.pointer );

    // allow zoom gestures if there is more than one pointer down
    if ( this.downPointers.length > 1 ) {
      this.downPointers.forEach( downPointer => downPointer.removeIntent( Pointer.Intent.DRAG ) );
      event.pointer.removeIntent( Pointer.Intent.DRAG );
    }

    if ( this._pointer === null && event.pointer.type === 'touch' ) {
      this._pointer = event.pointer;
      event.pointer.addInputListener( this._pointerListener );

      this.downPoint = event.pointer.point;
      this.currentPoint = this.downPoint.copy();
      this.previousPoint = this.currentPoint.copy();
    }
  }

  /**
   * @public
   * @param event
   */
  up( event ) {
    const index = this.downPointers.indexOf( event.pointer );
    if ( index > -1 ) {
      this.downPointers.splice( index, 1 );
    }
  }

  /**
   * Step the listener, updating timers used to determine swipe speeds and
   * double tap gestures.
   * @param dt
   * @private
   */
  step( dt ) {

    // detecting a double-tap
    if ( this.firstUp ) {
      this.timeSinceLastDown += dt;

      // too long for gesture, wait till next attempt
      if ( this.timeSinceLastDown > DOUBLE_TAP_INTERVAL ) {
        this.firstUp = false;
        this.timeSinceLastDown = 0;
      }
    }

    // detecting a press and hold
    if ( this._pointer && !this._pointer.isAttached() ) {
      this.holdingTime += dt;
      if ( this.holdingTime > PRESS_AND_HOLD_INTERVAL ) {

        // user has pressed down for long enough to forward a drag event to the
        // focused node
        const focusedNode = Display.focusedNode;
        if ( focusedNode && !this._pointer.listeners.includes( this._attachedPointerListener ) ) {

          // remove the unattached listener looking for gestures
          this.endSwipe();

          this.focusedNode = focusedNode;
          this._pointer.addInputListener( this._attachedPointerListener, true );

          this.focusedNode.swipeStart && this.focusedNode.swipeStart();
        }
      }
    }

    // determining swipe velocity
    if ( this.lastPoint !== null && this.currentPoint !== null ) {
      this.velocity = this.lastPoint.minus( this.currentPoint ).dividedScalar( dt );
    }
  }

  /**
   * Ends a swipe gesture, removing listeners and clearing references.
   * @private
   */
  endSwipe() {
    this.holdingTime = 0;

    this._pointer.removeInputListener( this._pointerListener );
  }

  /**
   * Interrupt this listener.
   * @public
   */
  interrupt() {
    this.endSwipe();
    this._pointer = null;
  }
}

scenery.register( 'SwipeListener', SwipeListener );
export default SwipeListener;