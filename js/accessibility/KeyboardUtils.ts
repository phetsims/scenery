// Copyright 2017-2023, University of Colorado Boulder

/**
 * Collection of utility constants and functions for managing keyboard input. Constants are values of Event.code, as
 * well as helper functions and collections.
 *
 * @author Jesse Greenberg
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import validate from '../../../axon/js/validate.js';
import { scenery } from '../imports.js';

const KEY_RIGHT_ARROW = 'ArrowRight';
const KEY_LEFT_ARROW = 'ArrowLeft';
const KEY_UP_ARROW = 'ArrowUp';
const KEY_DOWN_ARROW = 'ArrowDown';
const KEY_SHIFT_RIGHT = 'ShiftRight';
const KEY_SHIFT_LEFT = 'ShiftLeft';
const KEY_CONTROL_LEFT = 'ControlLeft';
const KEY_CONTROL_RIGHT = 'ControlRight';
const KEY_ALT_LEFT = 'AltLeft';
const KEY_ALT_RIGHT = 'AltRight';
const KEY_W = 'KeyW';
const KEY_A = 'KeyA';
const KEY_S = 'KeyS';
const KEY_D = 'KeyD';
const KEY_0 = 'Digit0';
const KEY_1 = 'Digit1';
const KEY_2 = 'Digit2';
const KEY_3 = 'Digit3';
const KEY_4 = 'Digit4';
const KEY_5 = 'Digit5';
const KEY_6 = 'Digit6';
const KEY_7 = 'Digit7';
const KEY_8 = 'Digit8';
const KEY_9 = 'Digit9';

const ARROW_KEYS = [ KEY_RIGHT_ARROW, KEY_LEFT_ARROW, KEY_UP_ARROW, KEY_DOWN_ARROW ];
const WASD_KEYS = [ KEY_W, KEY_S, KEY_A, KEY_D ];
const NUMBER_KEYS = [ KEY_0, KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6, KEY_7, KEY_8, KEY_9 ];
const SHIFT_KEYS = [ KEY_SHIFT_LEFT, KEY_SHIFT_RIGHT ];
const CONTROL_KEYS = [ KEY_CONTROL_LEFT, KEY_CONTROL_RIGHT ];
const ALT_KEYS = [ KEY_ALT_LEFT, KEY_ALT_RIGHT ];

const DOM_EVENT_VALIDATOR = { valueType: Event };
const ALL_KEYS: string[] = [];

/**
 * @extends {Object}
 * @type {{KEY_A: string, KEY_C: string, KEY_DOWN_ARROW: string, KEY_DELETE: string, KEY_B: string, KEY_E: string, KEY_D: string, KEY_G: string, KEY_F: string, KEY_I: string, KEY_H: string, KEY_K: string, KEY_RIGHT_ARROW: string, KEY_J: string, isArrowKey(Event): boolean, KEY_M: string, KEY_L: string, KEY_O: string, KEY_N: string, KEY_Q: string, KEY_P: string, KEY_S: string, KEY_BACKSPACE: string, KEY_R: string, KEY_U: string, KEY_MINUS: string, KEY_T: string, isMovementKey(Event): boolean, KEY_W: string, KEY_V: string, CONTROL_KEYS: (string)[], KEY_Y: string, KEY_X: string, getNumberFromCode(Event): (number|null), isAltKey(KeyboardEvent): *, KEY_Z: string, ARROW_KEYS: (string)[], KEY_SHIFT_LEFT: string, KEY_HOME: string, KEY_ESCAPE: string, isShiftKey(KeyboardEvent): *, KEY_ALT_LEFT: string, KEY_LEFT_ARROW: string, KEY_TAB: string, isAnyKeyEvent(x:Event, y:Array<string>): boolean, getEventCode(Event): (string|null), KEY_EQUALS: string, KEY_PAGE_UP: string, KEY_ALT_RIGHT: string, isWASDKey(Event): boolean, ALT_KEYS: (string)[], KEY_PAGE_DOWN: string, isNumberKey(Event): boolean, SHIFT_KEYS: (string)[], KEY_PLUS: string, WASD_KEYS: (string)[], isControlKey(KeyboardEvent): *, KEY_SHIFT_RIGHT: string, isRangeKey(Event): boolean, KEY_CONTROL_RIGHT: string, KEY_ENTER: string, isKeyEvent(Event, string): boolean, KEY_UP_ARROW: string, KEY_1: string, KEY_0: string, KEY_CONTROL_LEFT: string, KEY_3: string, KEY_2: string, KEY_5: string, KEY_4: string, KEY_7: string, KEY_6: string, KEY_9: string, MOVEMENT_KEYS: (string)[], KEY_8: string, KEY_SPACE: string, KEY_END: string}}
 */
const KeyboardUtils = {
  KEY_SPACE: 'Space',
  KEY_ENTER: 'Enter',
  KEY_TAB: 'Tab',
  KEY_RIGHT_ARROW: KEY_RIGHT_ARROW,
  KEY_LEFT_ARROW: KEY_LEFT_ARROW,
  KEY_UP_ARROW: KEY_UP_ARROW,
  KEY_DOWN_ARROW: KEY_DOWN_ARROW,
  KEY_SHIFT_LEFT: KEY_SHIFT_LEFT,
  KEY_SHIFT_RIGHT: KEY_SHIFT_RIGHT,
  KEY_ALT_LEFT: KEY_ALT_LEFT,
  KEY_ALT_RIGHT: KEY_ALT_RIGHT,
  KEY_CONTROL_LEFT: KEY_CONTROL_LEFT,
  KEY_CONTROL_RIGHT: KEY_CONTROL_RIGHT,
  KEY_ESCAPE: 'Escape',
  KEY_DELETE: 'Delete',
  KEY_BACKSPACE: 'Backspace',
  KEY_PAGE_UP: 'PageUp',
  KEY_PAGE_DOWN: 'PageDown',
  KEY_END: 'End',
  KEY_HOME: 'Home',
  KEY_0: KEY_0,
  KEY_1: KEY_1,
  KEY_2: KEY_2,
  KEY_3: KEY_3,
  KEY_4: KEY_4,
  KEY_5: KEY_5,
  KEY_6: KEY_6,
  KEY_7: KEY_7,
  KEY_8: KEY_8,
  KEY_9: KEY_9,
  KEY_A: 'KeyA',
  KEY_B: 'KeyB',
  KEY_C: 'KeyC',
  KEY_D: 'KeyD',
  KEY_E: 'KeyE',
  KEY_F: 'KeyF',
  KEY_G: 'KeyG',
  KEY_H: 'KeyH',
  KEY_I: 'KeyI',
  KEY_J: 'KeyJ',
  KEY_K: 'KeyK',
  KEY_L: 'KeyL',
  KEY_M: 'KeyM',
  KEY_N: 'KeyN',
  KEY_O: 'KeyO',
  KEY_P: 'KeyP',
  KEY_Q: 'KeyQ',
  KEY_R: 'KeyR',
  KEY_S: 'KeyS',
  KEY_T: 'KeyT',
  KEY_U: 'KeyU',
  KEY_V: 'KeyV',
  KEY_W: 'KeyW',
  KEY_X: 'KeyX',
  KEY_Y: 'KeyY',
  KEY_Z: 'KeyZ',

  // "Equals" and "Plus" share the same event.code, check for distinction with shift modifier key
  KEY_EQUALS: 'Equal',
  KEY_PLUS: 'Equal',
  KEY_MINUS: 'Minus',
  KEY_PERIOD: 'Period',

  ARROW_KEYS: ARROW_KEYS,
  WASD_KEYS: WASD_KEYS,
  MOVEMENT_KEYS: ARROW_KEYS.concat( WASD_KEYS ),
  SHIFT_KEYS: SHIFT_KEYS,
  CONTROL_KEYS: CONTROL_KEYS,
  ALT_KEYS: ALT_KEYS,

  /**
   * Returns whether the key corresponds to pressing an arrow key
   */
  isArrowKey( domEvent: Event | null ): boolean {
    return KeyboardUtils.isAnyKeyEvent( domEvent, ARROW_KEYS );
  },

  /**
   * Returns true if key is one of keys used for range inputs
   */
  isRangeKey( domEvent: Event | null ): boolean {
    return KeyboardUtils.isArrowKey( domEvent ) ||
           KeyboardUtils.isAnyKeyEvent( domEvent, [
             KeyboardUtils.KEY_PAGE_UP,
             KeyboardUtils.KEY_PAGE_DOWN,
             KeyboardUtils.KEY_HOME,
             KeyboardUtils.KEY_END
           ] );
  },

  /**
   * Returns whether or not the key corresponds to pressing a number key
   */
  isNumberKey( domEvent: Event | null ): boolean {
    return KeyboardUtils.isAnyKeyEvent( domEvent, NUMBER_KEYS );
  },

  /**
   * For number keys, return the number of the key, or null if not a number
   */
  getNumberFromCode( domEvent: Event | null ): null | number {
    if ( KeyboardUtils.isNumberKey( domEvent ) && domEvent instanceof KeyboardEvent ) {
      return Number( domEvent.code.replace( 'Digit', '' ) );
    }
    return null;
  },

  /**
   * Event.code distinguishes between left and right shift keys. If all you care about is the presence
   * of a shift key you can use this.
   */
  isShiftKey( domEvent: Event | null ): boolean {
    return KeyboardUtils.isAnyKeyEvent( domEvent, SHIFT_KEYS );
  },

  /**
   * Event.code distinguishes between left and right alt keys. If all you care about is the presence
   * of the alt key you can use this.
   */
  isAltKey( domEvent: Event | null ): boolean {
    return KeyboardUtils.isAnyKeyEvent( domEvent, ALT_KEYS );
  },

  /**
   * Event.code distinguishes between left and right control keys. If all you care about is the presence
   * of a control key you can use this.
   */
  isControlKey( domEvent: Event | null ): boolean {
    return KeyboardUtils.isAnyKeyEvent( domEvent, CONTROL_KEYS );
  },

  /**
   * Returns whether or not the key corresponds to one of the WASD movement keys.
   */
  isWASDKey( domEvent: Event | null ): boolean {
    return KeyboardUtils.isAnyKeyEvent( domEvent, WASD_KEYS );
  },

  /**
   * Returns true if the key indicates a 'movement' key in keyboard dragging
   */
  isMovementKey( domEvent: Event | null ): boolean {
    return KeyboardUtils.isAnyKeyEvent( domEvent, KeyboardUtils.MOVEMENT_KEYS );
  },

  /**
   * If the domEvent corresponds to any of the provided keys in the list.
   */
  isAnyKeyEvent( domEvent: Event | null, keyboardUtilsKeys: string[] ): boolean {
    validate( domEvent, DOM_EVENT_VALIDATOR );
    const code = KeyboardUtils.getEventCode( domEvent );
    return code ? keyboardUtilsKeys.includes( code ) : false;
  },

  /**
   * Whether or not the event was of the provided KeyboardUtils string.
   */
  isKeyEvent( domEvent: Event | null, keyboardUtilsKey: string ): boolean {
    return KeyboardUtils.getEventCode( domEvent ) === keyboardUtilsKey;
  },

  /**
   * Returns a string with the event.code that can be used to determine the keyboard keys of a KeyboardEvent object.
   * Otherwise, returns null if there is no code on the event. An example usage might look like
   *
   *  const key = KeyboardUtils.getEventCode( domEvent );
   *  if ( key === KeyboardUtils.KEY_A ) {
   *    // You pressed the A key!
   *  }
   *
   * @returns - null if there is no `code` property on the provided Event.
   */
  getEventCode( domEvent: Event | null ): string | null {
    let eventCode = null;
    if ( domEvent instanceof KeyboardEvent && domEvent.code ) {
      eventCode = domEvent.code;

      // If the Event is from the Numpad, the Event.key matches the desired Event.code when Num Lock is off. This
      // supports using the Num Pad for arrow keys and home/end/page up/page down.
      if ( eventCode.startsWith( 'Numpad' ) ) {
        eventCode = domEvent.key;
      }
    }

    return eventCode;
  },

  ALL_KEYS: ALL_KEYS
};

for ( const keyKey in KeyboardUtils ) {

  // @ts-expect-error No functions or key-groups allowed
  if ( KeyboardUtils.hasOwnProperty( keyKey ) && typeof KeyboardUtils[ keyKey ] === 'string' ) {

    // @ts-expect-error
    ALL_KEYS.push( KeyboardUtils[ keyKey ] as string );
  }
}

scenery.register( 'KeyboardUtils', KeyboardUtils );

export default KeyboardUtils;