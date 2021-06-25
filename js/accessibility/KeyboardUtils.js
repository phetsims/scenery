// Copyright 2017-2021, University of Colorado Boulder

/**
 * Collection of utility constants and functions for managing keyboard input. Constants are values of Event.key, as
 * well as helper functions and collections.
 *
 * @author Jesse Greenberg
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import validate from '../../../axon/js/validate.js';
import scenery from '../scenery.js';

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

// constants
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

  ARROW_KEYS: ARROW_KEYS,
  WASD_KEYS: WASD_KEYS,
  MOVEMENT_KEYS: ARROW_KEYS.concat( WASD_KEYS ),

  /**
   * Returns whether or not the key corresponds to pressing an arrow key
   * @public
   *
   * @param {Event} domEvent
   * @returns {boolean}
   */
  isArrowKey( domEvent ) {
    return KeyboardUtils.isAnyKeyEvent( domEvent, ARROW_KEYS );
  },

  /**
   * Returns true if key is one of keys used for range inputs
   * @public
   *
   * @param {Event} domEvent
   * @returns {boolean}
   */
  isRangeKey( domEvent ) {
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
   * @public
   *
   * @param {Event} domEvent
   * @returns {boolean}
   */
  isNumberKey( domEvent ) {
    return KeyboardUtils.isAnyKeyEvent( domEvent, NUMBER_KEYS );
  },

  /**
   * Event.code distinguishes between left and right shift keys. If all you care about is the presence
   * of a shift key you can use this.
   * @public
   *
   * @param domEvent
   * @returns {*}
   */
  isShiftKey( domEvent ) {
    return KeyboardUtils.isAnyKeyEvent( domEvent, SHIFT_KEYS );
  },

  /**
   * Event.code distinguishes between left and right alt keys. If all you care about is the presence
   * of the alt key you can use this.
   * @public
   *
   * @param domEvent
   * @returns {*}
   */
  isAltKey( domEvent ) {
    return KeyboardUtils.isAnyKeyEvent( domEvent, ALT_KEYS );
  },

  /**
   * Event.code distinguishes between left and right control keys. If all you care about is the presence
   * of a control key you can use this.
   * @public
   *
   * @param domEvent
   * @returns {*}
   */
  isControlKey( domEvent ) {
    return KeyboardUtils.isAnyKeyEvent( domEvent, CONTROL_KEYS );
  },

  /**
   * Returns whether or not the key corresponds to one of the WASD movement keys.
   * @public
   *
   * @param {Event} domEvent
   * @returns {boolean}
   */
  isWASDKey( domEvent ) {
    return KeyboardUtils.isAnyKeyEvent( domEvent, WASD_KEYS );
  },

  /**
   * Returns true if the key indicates a 'movement' key in keyboard dragging
   * @public
   *
   * @param {Event} domEvent
   * @returns {boolean}
   */
  isMovementKey( domEvent ) {
    return KeyboardUtils.isAnyKeyEvent( domEvent, KeyboardUtils.MOVEMENT_KEYS );
  },

  /**
   * If the domEvent corresponds to any of the provided keys in the list.
   * @public
   *
   * @param {Event} domEvent
   * @param {string[]} keyboardUtilsKeys
   * @returns {*}
   */
  isAnyKeyEvent( domEvent, keyboardUtilsKeys ) {
    validate( domEvent, DOM_EVENT_VALIDATOR );
    return keyboardUtilsKeys.includes( KeyboardUtils.getEventCode( domEvent ) );
  },

  /**
   * Whether or not the event was of the provided KeyboardUtils string.
   * @public
   *
   * @param {Event} domEvent
   * @param {string} keyboardUtilsKey
   * @returns {boolean}
   */
  isKeyEvent( domEvent, keyboardUtilsKey ) {
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
   * @public
   * @param {Event} domEvent
   * @returns {string|null} - null if there is no `key` property on the provided Event.
   */
  getEventCode( domEvent ) {
    validate( domEvent, DOM_EVENT_VALIDATOR );
    return domEvent.code ? domEvent.code : null;
  }
};

const ALL_KEYS = [];
for ( const keyKey in KeyboardUtils ) {

  // No functions or key-groups allowed
  if ( KeyboardUtils.hasOwnProperty( keyKey ) && typeof KeyboardUtils[ keyKey ] === 'string' ) {
    ALL_KEYS.push( KeyboardUtils[ keyKey ] );
  }
}

// @public - Not really all of them, but all that are in the above list. If you see one you wish was in here, then add it!
KeyboardUtils.ALL_KEYS = ALL_KEYS;

scenery.register( 'KeyboardUtils', KeyboardUtils );

export default KeyboardUtils;