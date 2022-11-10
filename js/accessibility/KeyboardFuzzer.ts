// Copyright 2018-2022, University of Colorado Boulder

/**
 * ?fuzzBoard keyboard fuzzer
 * TODO: keep track of keyState so that we don't trigger a keydown of keyA before the previous keyA keyup event has been called.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import stepTimer from '../../../axon/js/stepTimer.js';
import { TimerListener } from '../../../axon/js/Timer.js';
import Random from '../../../dot/js/Random.js';
import { Display, globalKeyStateTracker, KeyboardUtils, PDOMUtils, scenery } from '../imports.js';

type KeyupListener = ( () => void ) & {
  timeout: TimerListener;
};


// uppercase matters
const keyboardTestingSchema: Record<string, string[]> = {
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

class KeyboardFuzzer {
  private readonly display: Display;
  private readonly random: Random;
  private readonly numberOfComponentsTested: number;
  private keyupListeners: KeyupListener[];
  private currentElement: Element | null;

  public constructor( display: Display, seed: number ) {

    this.display = display;
    this.random = new Random( { seed: seed } );
    this.numberOfComponentsTested = 10;
    this.keyupListeners = [];
    this.currentElement = null;
  }

  /**
   * Randomly decide if we should focus the next element, or stay focused on the current element
   */
  private chooseNextElement(): void {
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

  private clearListeners(): void {
    this.keyupListeners.forEach( listener => {
      assert && assert( typeof listener.timeout === 'function', 'should have an attached timeout' );
      stepTimer.clearTimeout( listener.timeout );
      listener();
      assert && assert( !this.keyupListeners.includes( listener ), 'calling listener should remove itself from the keyupListeners.' );
    } );
  }

  private triggerClickEvent(): void {
    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( 'triggering click' );
    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();

    // We'll only ever want to send events to the activeElement (so that it's not stale), see
    // https://github.com/phetsims/scenery/issues/1497
    const element = document.activeElement;
    element instanceof HTMLElement && element.click();

    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
  }

  /**
   * Trigger a keydown/keyup pair. The keyup is triggered with a timeout.
   */
  private triggerKeyDownUpEvents( code: string ): void {

    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( `trigger keydown/up: ${code}` );
    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();

    // TODO: screen readers normally take our keydown events, but may not here, is the discrepancy ok?
    this.triggerDOMEvent( KEY_DOWN, code );

    const randomTimeForKeypress = this.random.nextInt( MAX_MS_KEY_HOLD_DOWN );

    const keyupListener: KeyupListener = () => {
      this.triggerDOMEvent( KEY_UP, code );
      if ( this.keyupListeners.includes( keyupListener ) ) {
        this.keyupListeners.splice( this.keyupListeners.indexOf( keyupListener ), 1 );
      }
    };

    keyupListener.timeout = stepTimer.setTimeout( keyupListener, randomTimeForKeypress === MAX_MS_KEY_HOLD_DOWN ? 2000 : randomTimeForKeypress );
    this.keyupListeners.push( keyupListener );

    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
  }

  /**
   * Trigger a keydown/keyup pair with a random KeyboardEvent.code.
   */
  private triggerRandomKeyDownUpEvents( element: Element ): void {

    const randomCode = ALL_KEYS[ Math.floor( this.random.nextDouble() * ( ALL_KEYS.length - 1 ) ) ];

    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( `trigger random keydown/up: ${randomCode}` );
    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();

    this.triggerKeyDownUpEvents( randomCode );

    sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
  }

  /**
   * A random event creator that sends keyboard events. Based on the idea of fuzzMouse, but to test/spam accessibility
   * related keyboard navigation and alternate input implementation.
   *
   * TODO: NOTE: Right now this is a very experimental implementation. Tread wearily
   * TODO: @param keyboardPressesPerFocusedItem {number} - basically would be the same as fuzzRate, but handling
   * TODO:     the keydown events for a focused item
   */
  public fuzzBoardEvents( fuzzRate: number ): void {

    if ( this.display && this.display._input && this.display._input.pdomPointer ) {
      const pdomPointer = this.display._input.pdomPointer;
      if ( pdomPointer && !pdomPointer.blockTrustedEvents ) {
        pdomPointer.blockTrustedEvents = true;
      }
    }

    for ( let i = 0; i < this.numberOfComponentsTested; i++ ) {

      // find a focus a random element
      this.chooseNextElement();

      for ( let i = 0; i < fuzzRate / this.numberOfComponentsTested; i++ ) {

        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.KeyboardFuzzer( `main loop, i=${i}` );
        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.push();

        // get active element, focus might have changed in the last press
        const elementWithFocus = document.activeElement;

        if ( elementWithFocus && keyboardTestingSchema[ elementWithFocus.tagName.toUpperCase() ] ) {

          const randomNumber = this.random.nextDouble();
          if ( randomNumber < DO_KNOWN_KEYS_THRESHOLD ) {
            const codeValues = keyboardTestingSchema[ elementWithFocus.tagName ];
            const code = this.random.sample( codeValues );
            this.triggerKeyDownUpEvents( code );
          }
          else if ( randomNumber < CLICK_EVENT_THRESHOLD ) {
            this.triggerClickEvent();
          }
          else {
            this.triggerRandomKeyDownUpEvents( elementWithFocus );
          }
        }
        else {
          elementWithFocus && this.triggerRandomKeyDownUpEvents( elementWithFocus );
        }
        // TODO: What about other types of events, not just keydown/keyup??!?!
        // TODO: what about application role elements

        sceneryLog && sceneryLog.KeyboardFuzzer && sceneryLog.pop();
      }
    }
  }

  /**
   * Taken from example in http://output.jsbin.com/awenaq/3,
   */
  private triggerDOMEvent( event: string, code: string ): void {
    // We'll only ever want to send events to the activeElement (so that it's not stale), see
    // https://github.com/phetsims/scenery/issues/1497
    if ( document.activeElement ) {
      const eventObj = new KeyboardEvent( event, {
        bubbles: true,
        code: code,
        shiftKey: globalKeyStateTracker.shiftKeyDown,
        altKey: globalKeyStateTracker.altKeyDown,
        ctrlKey: globalKeyStateTracker.ctrlKeyDown
      } );

      document.activeElement.dispatchEvent( eventObj );
    }
  }
}

scenery.register( 'KeyboardFuzzer', KeyboardFuzzer );
export default KeyboardFuzzer;