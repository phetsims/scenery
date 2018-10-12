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
  const KeyboardUtil = require( 'SCENERY/accessibility/KeyboardUtil' );
  const Random = require( 'DOT/Random' );
  const scenery = require( 'SCENERY/scenery' );
  const timer = require( 'PHET_CORE/timer' );

  // uppercase matters
  const keyboardTestingSchema = {
    INPUT: [ ...KeyboardUtil.ARROW_KEYS, KeyboardUtil.KEY_PAGE_UP, KeyboardUtil.KEY_PAGE_DOWN,
      KeyboardUtil.KEY_HOME, KeyboardUtil.KEY_END ],
    DIV: [ KeyboardUtil.KEY_ESCAPE ],
    P: [ KeyboardUtil.KEY_ESCAPE ],
    BUTTON: [ KeyboardUtil.KEY_ENTER, KeyboardUtil.KEY_SPACE ]
  };

  const NEXT_ELEMENT_THRESHOLD = .10;
  const DO_KNOWN_KEYS_THRESHOLD = .75;

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

      this.keysPressedEachFrame = 10;

      // {HTMLElement}
      this.currentElement = null;
    }

    /**
     * Randomly decide if we should focus the next element, or stay focused on the current element
     */
    chooseNextElement() {
      if ( this.currentElement === null ) {
        this.currentElement = document.activeElement;
      }
      else if ( this.random.nextDouble() < NEXT_ELEMENT_THRESHOLD ) {
        var nextFocusable = AccessibilityUtil.getRandomFocusable();
        nextFocusable.focus();
        this.currentElement = nextFocusable;
      }

    }

    /**
     * Trigger a keydown/keyup pair. The keyup is triggered with a timeout
     * @param {HTMLElement} element
     * @param {number} keyCode
     */
    triggerKeyDownUpEvents( element, keyCode ) {
      // TODO: screen readers normally take our keydown events, but may not here, is the descrpency ok?
      KeyboardFuzzer.triggerDOMEvent( 'keydown', element, keyCode );

      timer.setTimeout( () => {
        KeyboardFuzzer.triggerDOMEvent( 'keyup', element, keyCode );

      }, 1 ); // TODO: make this time variable?
    }

    /**
     * Trigger a keydown/keyup pair with a random keyCode
     * @param {HTMLElement} element
     */
    triggerRandomKeyDownUpEvents( element ) {

      var randomKeyCode = Math.floor( this.random.nextDouble() * ( max - min ) + min );

      this.triggerKeyDownUpEvents( element, randomKeyCode );

    }

    /**
     * A random event creater that sends keyboard events. Based on the idea of fuzzMouse, but to test/spam accessibility
     * related keyboard navigation and alternate input implementation.
     *
     * TODO: NOTE: Right now this is a very experimental implementation. Tread wearily
     * TODO: @param keyboardPressesPerFocusedItem {number} - basically would be the same as fuzzRate, but handling
     * TODO:     the keydown events for a focused item
     */
    fuzzBoardEvents() {

      this.chooseNextElement();

      var elementWithFocus = document.activeElement;

      for ( let i = 0; i < this.keysPressedEachFrame; i++ ) {

        if ( keyboardTestingSchema[ elementWithFocus.tagName ] ) {

          if ( this.random.nextDouble() > DO_KNOWN_KEYS_THRESHOLD ) {
            const keyCodes = keyboardTestingSchema[ elementWithFocus.tagName ];
            const keyCode = this.random.sample( keyCodes );
            this.triggerKeyDownUpEvents( elementWithFocus, keyCode );
          }
          else {
            this.triggerRandomKeyDownUpEvents( elementWithFocus );
          }
        }
        // TODO: What about other types of events, not just keydown/keyup??!?!
        // TODO: what about application role elements
      }
    }

    /**
     * Taken from example in http://output.jsbin.com/awenaq/3,
     * @param {string} event
     * @param {HTMLElement} element
     * @param {number} [keycode]
     */
    static triggerDOMEvent( event, element, keyCode ) {
      var eventObj = document.createEventObject ?
                     document.createEventObject() : document.createEvent( 'Events' );

      if ( eventObj.initEvent ) {
        eventObj.initEvent( event, true, true );
      }

      eventObj.keyCode = keyCode;
      // eventObj.shiftKey = true; // TODO: we can add modifier keys in here with options?
      eventObj.which = keyCode;

      element.dispatchEvent ? element.dispatchEvent( eventObj ) : element.fireEvent( 'on' + event, eventObj );
    }
  }

  return scenery.register( 'KeyboardFuzzer', KeyboardFuzzer );
} );