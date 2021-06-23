// Copyright 2018-2021, University of Colorado Boulder

/**
 * ?fuzzBoard keyboard fuzzer
 * TODO: keep track of keyState so that we don't trigger a keydown of keyA before the previous keyA keyup event has been called.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import stepTimer from '../../../axon/js/stepTimer.js';
import Random from '../../../dot/js/Random.js';
import scenery from '../scenery.js';
import globalKeyStateTracker from './globalKeyStateTracker.js';
import KeyboardUtils from './KeyboardUtils.js';
import PDOMUtils from './pdom/PDOMUtils.js';

// uppercase matters
const keyboardTestingSchema = {
  INPUT: [ ...KeyboardUtils.ARROW_KEYS, KeyboardUtils.KEY_PAGE_UP, KeyboardUtils.KEY_PAGE_DOWN,
    KeyboardUtils.KEY_HOME, KeyboardUtils.KEY_END, KeyboardUtils.KEY_ENTER, KeyboardUtils.KEY_SPACE ],
  DIV: [ ...KeyboardUtils.ARROW_KEYS, ...KeyboardUtils.WASD_KEYS ],
  P: [ KeyboardUtils.KEY_ESCAPE ],
  BUTTON: [ KeyboardUtils.KEY_ENTER, KeyboardUtils.KEY_SPACE ]
};

const ALL_KEYS = KeyboardUtils.ALL_KEYS;

const MAX_MS_KEY_HOLD_DOWN = 100;
const NEXT_ELEMENT_THRESHOLD = 0.1;

const DO_KNOWN_KEYS_THRESHOLD = 0.60; // for keydown/up, 60 percent of the events
const CLICK_EVENT_THRESHOLD = DO_KNOWN_KEYS_THRESHOLD + 0.10; // 10 percent of the events

const KEY_DOWN = 'keydown';
const KEY_UP = 'keyup';

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
    this.numberOfComponentsTested = 10;
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
      const nextFocusable = PDOMUtils.getRandomFocusable( this.random );
      nextFocusable.focus();
      this.currentElement = nextFocusable;

      sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
    }
  }

  /**
   * @private
   */
  clearListeners() {
    this.keyupListeners.forEach( listener => {
      assert && assert( typeof listener.timeout === 'function', 'should have an attached timeout' );
      stepTimer.clearTimeout( listener.timeout );
      listener();
      assert && assert( !this.keyupListeners.includes( listener ), 'calling listener should remove itself from the keyupListeners.' );
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
   * Trigger a keydown/keyup pair. The keyup is triggered with a timeout.
   * @private
   *
   * @param {HTMLElement} element
   * @param {string} key
   */
  triggerKeyDownUpEvents( element, key ) {

    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( `trigger keydown/up: ${key}` );
    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();

    // TODO: screen readers normally take our keydown events, but may not here, is the discrepancy ok?
    this.triggerDOMEvent( KEY_DOWN, element, key );

    const randomTimeForKeypress = this.random.nextInt( MAX_MS_KEY_HOLD_DOWN );

    const keyupListener = () => {
      this.triggerDOMEvent( KEY_UP, element, key );
      if ( this.keyupListeners.includes( keyupListener ) ) {
        this.keyupListeners.splice( this.keyupListeners.indexOf( keyupListener ), 1 );
      }
    };

    keyupListener.timeout = stepTimer.setTimeout( keyupListener, randomTimeForKeypress === MAX_MS_KEY_HOLD_DOWN ? 2000 : randomTimeForKeypress );
    this.keyupListeners.push( keyupListener );

    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
  }

  /**
   * Trigger a keydown/keyup pair with a random key
   * @private
   * @param {HTMLElement} element
   */
  triggerRandomKeyDownUpEvents( element ) {

    const randomKey = ALL_KEYS[ Math.floor( this.random.nextDouble() * ( ALL_KEYS.length - 1 ) ) ];

    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( `trigger random keydown/up: ${randomKey}` );
    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();

    this.triggerKeyDownUpEvents( element, randomKey );

    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
  }

  /**
   * A random event creator that sends keyboard events. Based on the idea of fuzzMouse, but to test/spam accessibility
   * related keyboard navigation and alternate input implementation.
   *
   * @public
   * TODO: NOTE: Right now this is a very experimental implementation. Tread wearily
   * TODO: @param keyboardPressesPerFocusedItem {number} - basically would be the same as fuzzRate, but handling
   * TODO:     the keydown events for a focused item
   */
  fuzzBoardEvents( fuzzRate ) {

    const pdomPointer = this.display._input.pdomPointer;
    if ( pdomPointer && !pdomPointer.blockTrustedEvents ) {
      pdomPointer.blockTrustedEvents = true;
    }

    for ( let i = 0; i < this.numberOfComponentsTested; i++ ) {

      // find a focus a random element
      this.chooseNextElement();

      for ( let i = 0; i < fuzzRate / this.numberOfComponentsTested; i++ ) {

        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( `main loop, i=${i}` );
        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();

        // get active element, focus might have changed in the last press
        const elementWithFocus = document.activeElement;

        if ( keyboardTestingSchema[ elementWithFocus.tagName.toUpperCase() ] ) {

          const randomNumber = this.random.nextDouble();
          if ( randomNumber < DO_KNOWN_KEYS_THRESHOLD ) {
            const keyValues = keyboardTestingSchema[ elementWithFocus.tagName ];
            const key = this.random.sample( keyValues );
            this.triggerKeyDownUpEvents( elementWithFocus, key );
          }
          else if ( randomNumber < CLICK_EVENT_THRESHOLD ) {
            this.triggerClickEvent( elementWithFocus );
          }
          else {
            this.triggerRandomKeyDownUpEvents( elementWithFocus );
          }
        }
        else {
          this.triggerRandomKeyDownUpEvents( elementWithFocus );
        }
        // TODO: What about other types of events, not just keydown/keyup??!?!
        // TODO: what about application role elements

        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
      }
    }
  }

  /**
   * Taken from example in http://output.jsbin.com/awenaq/3,
   * @param {string} event
   * @param {HTMLElement} element
   * @param {string} key
   * @private
   */
  triggerDOMEvent( event, element, key ) {
    const eventObj = new KeyboardEvent( event, {
      bubbles: true,
      key: key,
      shiftKey: globalKeyStateTracker.shiftKeyDown,
      altKey: globalKeyStateTracker.altKeyDown,
      ctrlKey: globalKeyStateTracker.ctrlKeyDown
    } );

    element.dispatchEvent( eventObj );
  }
}

scenery.register( 'KeyboardFuzzer', KeyboardFuzzer );
export default KeyboardFuzzer;