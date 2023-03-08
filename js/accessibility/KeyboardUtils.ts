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

// These are `KeyboardEvent.key` values! The difference is necessary because KeyboardEvent.code distinguishes between
// left and right, but we often need to watch for both at the same time.
const KEY_ALT = 'Alt';
const KEY_SHIFT = 'Shift';
const KEY_CONTROL = 'Control';

const ARROW_KEYS = [ KEY_RIGHT_ARROW, KEY_LEFT_ARROW, KEY_UP_ARROW, KEY_DOWN_ARROW ];
const WASD_KEYS = [ KEY_W, KEY_S, KEY_A, KEY_D ];
const NUMBER_KEYS = [ KEY_0, KEY_1, KEY_2, KEY_3, KEY_4, KEY_5, KEY_6, KEY_7, KEY_8, KEY_9 ];
const SHIFT_KEYS = [ KEY_SHIFT_LEFT, KEY_SHIFT_RIGHT ];
const CONTROL_KEYS = [ KEY_CONTROL_LEFT, KEY_CONTROL_RIGHT ];
const ALT_KEYS = [ KEY_ALT_LEFT, KEY_ALT_RIGHT ];

// These are KeyboardEvent.key values, excluding left/right KeyboardEvent.codes
const MODIFIER_KEYS = [ KEY_ALT, KEY_CONTROL, KEY_SHIFT ];

const DOM_EVENT_VALIDATOR = { valueType: Event };
const ALL_KEY_CODES: string[] = [];

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
  KEY_SHIFT: KEY_SHIFT,
  KEY_ALT: KEY_ALT,
  KEY_CONTROL: KEY_CONTROL,
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
  KEY_NUMPAD_0: 'Numpad0',
  KEY_NUMPAD_1: 'Numpad1',
  KEY_NUMPAD_2: 'Numpad2',
  KEY_NUMPAD_3: 'Numpad3',
  KEY_NUMPAD_4: 'Numpad4',
  KEY_NUMPAD_5: 'Numpad5',
  KEY_NUMPAD_6: 'Numpad6',
  KEY_NUMPAD_7: 'Numpad7',
  KEY_NUMPAD_8: 'Numpad8',
  KEY_NUMPAD_9: 'Numpad9',
  KEY_NUMPAD_DECIMAL: 'NumpadDecimal',
  KEY_NUMPAD_PLUS: 'NumpadAdd',
  KEY_NUMPAD_MINUS: 'NumpadSubtract',
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

  // Maps a KeyboardEvent.key to the left/right pair of KeyboardEvent.code for modifier keys
  MODIFIER_KEY_TO_CODE_MAP: new Map( [
    [ KEY_ALT, ALT_KEYS ],
    [ KEY_SHIFT, SHIFT_KEYS ],
    [ KEY_CONTROL, CONTROL_KEYS ]
  ] ),


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
   * Whether the event was of the provided KeyboardUtils string.
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

      // For Numpad keys, the DomEvent.code does not match the equivalent "normal" keyboard key, but the DomEvent.key
      // will match the code of the "normal" key. In those cases (home/page up/page down/end) use the key as the code.
      if ( eventCode.startsWith( 'Numpad' ) && ALL_KEY_CODES.includes( domEvent.key ) ) {
        eventCode = domEvent.key;
      }
    }

    return eventCode;
  },

  /**
   * Returns true when if the provided string is a KeyboardEvent.key for a modifier key. Note that is KeyboardEvent.key
   * **NOT** KeyboardEvent.code. KeyboardEvent.code does not distinguish between left and right modifier keys so we
   * have some special behavior when we detect a modifier key (left or right) is pressed.
   * @param key
   */
  isModifierKey( key: string ): boolean {
    return MODIFIER_KEYS.includes( key );
  },

  /**
   * Returns true if the provided KeyboardEvent.code/KeyboardEvent.key is equivalent to the provided KeyboardEvent.code.
   * Specifically comparing modifier keys. If both are `code`, returns true when they are equal. If first value is
   * a `key` for alt/control/shift modifier key, then it returns true when the code is one of the matching
   * left/right `codes` for that `key`. For example
   *
   * `keyOrCode` = 'Shift', `code` = 'ShiftLeft -> true
   * `keyOrCode = 'Alt', `code` = 'AltRight' -> true
   * `keyOrCode = 'Control`, `code` = 'KeyR' -> false
   *
   * @param keyOrCode - KeyboardEvent.key OR KeyboardEvent.code
   * @param code - KeyboardEvent.code
   */
  areKeysEquivalent( keyOrCode: string, code: string ): boolean {
    const equivalentModifierKeys = KeyboardUtils.MODIFIER_KEY_TO_CODE_MAP.get( keyOrCode );
    if ( equivalentModifierKeys ) {
      return equivalentModifierKeys.includes( code );
    }
    else {
      return keyOrCode === code;
    }
  },

  ALL_KEYS: ALL_KEY_CODES
};

for ( const keyKey in KeyboardUtils ) {

  // @ts-expect-error No functions or key-groups allowed
  if ( KeyboardUtils.hasOwnProperty( keyKey ) && typeof KeyboardUtils[ keyKey ] === 'string' ) {

    // @ts-expect-error
    ALL_KEY_CODES.push( KeyboardUtils[ keyKey ] as string );
  }
}

scenery.register( 'KeyboardUtils', KeyboardUtils );

export default KeyboardUtils;