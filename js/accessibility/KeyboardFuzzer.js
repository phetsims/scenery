// Copyright 2018, University of Colorado Boulder

/**
 * ?fuzzBoard keyboard fuzzer
 * TODO: keep track of keystate so that we don't trigger a keydown of keyA before the previous keyA keyup event has been called.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */
define( require => {
  'use strict';

  // modules
  const AccessibilityUtil = require( 'SCENERY/accessibility/AccessibilityUtil' );
  const Display = require( 'SCENERY/display/Display' );
  const KeyboardUtil = require( 'SCENERY/accessibility/KeyboardUtil' );
  const Random = require( 'DOT/Random' );
  const scenery = require( 'SCENERY/scenery' );
  const timer = require( 'AXON/timer' );

  // uppercase matters
  const keyboardTestingSchema = {
    INPUT: [ ...KeyboardUtil.ARROW_KEYS, KeyboardUtil.KEY_PAGE_UP, KeyboardUtil.KEY_PAGE_DOWN,
      KeyboardUtil.KEY_HOME, KeyboardUtil.KEY_END, KeyboardUtil.KEY_ENTER, KeyboardUtil.KEY_SPACE ],
    DIV: [ ...KeyboardUtil.ARROW_KEYS ],
    P: [ KeyboardUtil.KEY_ESCAPE ],
    BUTTON: [ KeyboardUtil.KEY_ENTER, KeyboardUtil.KEY_SPACE ]
  };

  const MAX_MS_KEY_HOLD_DOWN = 200;
  const NEXT_ELEMENT_THRESHOLD = .1;

  const DO_KNOWN_KEYS_THRESHOLD = .60; // for keydown/up
  const CLICK_EVENT = .10; // TODO because of implementation this is actually 4%. but reads like "10% of the time after 60% of the time"

  const KEY_DOWN = 'keydown';
  const KEY_UP = 'keyup';

  var min = 9;
  var max = 223;

  /**
   *
   * @param {Object} [options]
   * @constructor
   */
  class KeyboardFuzzer {
    constructor( display, seed ) {


      // @private
      this.display = display;
      this.random = new Random( { seed: seed } );
      this.keysPressedEachFrame = 1;
      this.keyupListeners = [];

      // @private {HTMLElement}
      this.currentElement = null;
    }

    /**
     * @private
     * Randomly decide if we should focus the next element, or stay focused on the current element
     */
    chooseNextElement() {
      if ( this.currentElement === null ) {
        this.currentElement = document.activeElement;
      }
      else if ( this.random.nextDouble() < NEXT_ELEMENT_THRESHOLD ) {
        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( 'choosing new element' );
        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();

        // before we change focus to the next item, immediately release all keys that were down on the active element
        this.clearListeners();
        var nextFocusable = AccessibilityUtil.getRandomFocusable( this.random );
        nextFocusable.focus();
        this.currentElement = nextFocusable;

        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
      }
    }

    /**
     * @private
     */
    clearListeners() {
      this.keyupListeners.forEach( function( listener ) {
        timer.clearTimeout( listener );
        listener();
      } );
    }

    /**
     * @private
     * @param {HTMLElement} element
     */
    triggerClickEvent( element ) {
      sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( 'triggering click' );
      sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();

      element.click();

      sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
    }

    /**
     * Trigger a keydown/keyup pair. The keyup is triggered with a timeout
     * @param {HTMLElement} element
     * @param {number} keyCode
     */
    triggerKeyDownUpEvents( element, keyCode ) {

      if ( !Display.keyStateTracker.isKeyDown( keyCode ) ) {
        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( 'trigger keydown/up: ' + keyCode );
        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();


        // TODO: screen readers normally take our keydown events, but may not here, is the descrpency ok?
        this.triggerDOMEvent( KEY_DOWN, element, keyCode );

        this.keyupListeners.push( timer.setTimeout( () => {
            this.triggerDOMEvent( KEY_UP, element, keyCode );

        }, this.random.nextInt( MAX_MS_KEY_HOLD_DOWN ) ) );

        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
      }
    }

    /**
     * Trigger a keydown/keyup pair with a random keyCode
     * @private
     * @param {HTMLElement} element
     */
    triggerRandomKeyDownUpEvents( element ) {

      var randomKeyCode = Math.floor( this.random.nextDouble() * ( max - min ) + min );

      sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( 'trigger random keydown/up: ' + randomKeyCode );
      sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();

      this.triggerKeyDownUpEvents( element, randomKeyCode );

      sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
    }

    /**
     * A random event creater that sends keyboard events. Based on the idea of fuzzMouse, but to test/spam accessibility
     * related keyboard navigation and alternate input implementation.
     *
     * @public
     * TODO: NOTE: Right now this is a very experimental implementation. Tread wearily
     * TODO: @param keyboardPressesPerFocusedItem {number} - basically would be the same as fuzzRate, but handling
     * TODO:     the keydown events for a focused item
     */
    fuzzBoardEvents() {

      const a11yPointer = this.display._input.a11yPointer;
      if ( a11yPointer && !a11yPointer.blockTrustedEvents) {
        a11yPointer.blockTrustedEvents = true;
      }

      this.chooseNextElement();

      for ( let i = 0; i < this.keysPressedEachFrame; i++ ) {

        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( 'main loop, i=' + i );
        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();


        // get active element, focus might have changed in the last press
        var elementWithFocus = document.activeElement;

        if ( keyboardTestingSchema[ elementWithFocus.tagName ] ) {

          if ( this.random.nextDouble() < DO_KNOWN_KEYS_THRESHOLD ) {
            const keyCodes = keyboardTestingSchema[ elementWithFocus.tagName ];
            const keyCode = this.random.sample( keyCodes );
            this.triggerKeyDownUpEvents( elementWithFocus, keyCode );
          }
          else if ( this.random.nextDouble() < CLICK_EVENT ) {
            this.triggerClickEvent( elementWithFocus );
          }
          else {
            this.triggerRandomKeyDownUpEvents( elementWithFocus );
          }
        }
        // TODO: What about other types of events, not just keydown/keyup??!?!
        // TODO: what about application role elements

        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
      }
    }

    /**
     * Taken from example in http://output.jsbin.com/awenaq/3,
     * @param {string} event
     * @param {HTMLElement} element
     * @param {number} [keycode]
     * @private
     */
    triggerDOMEvent( event, element, keyCode ) {
      var eventObj = document.createEventObject ?
                     document.createEventObject() : document.createEvent( 'Events' );

      if ( eventObj.initEvent ) {
        eventObj.initEvent( event, true, true );
      }

      eventObj.keyCode = keyCode;
      // eventObj.shiftKey = true; // TODO: we can add modifier keys in here with options?
      eventObj.which = keyCode;

      // add any modifier keys to the event
      if ( Display.keyStateTracker.shiftKeyDown ) {
        eventObj.shiftKey = true;
      }
      if ( Display.keyStateTracker.altKeyDown ) {
        eventObj.altKey = true;
      }
      if ( Display.keyStateTracker.ctrlKeyDown ) {
        eventObj.ctrlKey = true;
      }
      
      element.dispatchEvent ? element.dispatchEvent( eventObj ) : element.fireEvent( 'on' + event, eventObj );
    }
  }

  return scenery.register( 'KeyboardFuzzer', KeyboardFuzzer );
} );