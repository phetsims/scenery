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

const KEY_RIGHT_ARROW = 'arrowright';
const KEY_LEFT_ARROW = 'arrowleft';
const KEY_UP_ARROW = 'arrowup';
const KEY_DOWN_ARROW = 'arrowdown';
const KEY_SHIFT = 'shift';
const KEY_CTRL = 'control';
const KEY_ALT = 'alt';
const KEY_W = 'w';
const KEY_A = 'a';
const KEY_S = 's';
const KEY_D = 'd';

const ARROW_KEYS = [ KEY_RIGHT_ARROW, KEY_LEFT_ARROW, KEY_UP_ARROW, KEY_DOWN_ARROW ];
const WASD_KEYS = [ KEY_W, KEY_S, KEY_A, KEY_D ];

const DOM_EVENT_VALIDATOR = { valueType: Event };

// constants
var KeyboardUtils = {
  KEY_SPACE: ' ',
  KEY_ENTER: 'enter',
  KEY_TAB: 'tab',
  KEY_RIGHT_ARROW: KEY_RIGHT_ARROW,
  KEY_LEFT_ARROW: KEY_LEFT_ARROW,
  KEY_UP_ARROW: KEY_UP_ARROW,
  KEY_DOWN_ARROW: KEY_DOWN_ARROW,
  KEY_SHIFT: KEY_SHIFT,
  KEY_CTRL: KEY_CTRL,
  KEY_ALT: KEY_ALT,
  KEY_ESCAPE: 'escape',
  KEY_DELETE: 'delete',
  KEY_BACKSPACE: 'backspace',
  KEY_PAGE_UP: 'pageup',
  KEY_PAGE_DOWN: 'pagedown',
  KEY_END: 'end',
  KEY_HOME: 'home',
  KEY_0: '0',
  KEY_1: '1',
  KEY_2: '2',
  KEY_3: '3',
  KEY_4: '4',
  KEY_5: '5',
  KEY_6: '6',
  KEY_7: '7',
  KEY_8: '8',
  KEY_9: '9',
  KEY_A: 'a',
  KEY_B: 'b',
  KEY_C: 'c',
  KEY_D: 'd',
  KEY_E: 'e',
  KEY_F: 'f',
  KEY_G: 'g',
  KEY_H: 'h',
  KEY_I: 'I',
  KEY_J: 'j',
  KEY_K: 'k',
  KEY_L: 'l',
  KEY_M: 'm',
  KEY_N: 'n',
  KEY_O: 'o',
  KEY_P: 'p',
  KEY_Q: 'q',
  KEY_R: 'r',
  KEY_S: 's',
  KEY_T: 't',
  KEY_U: 'u',
  KEY_V: 'v',
  KEY_W: 'w',
  KEY_X: 'x',
  KEY_Y: 'y',
  KEY_Z: 'z',

  KEY_EQUALS: '=',
  KEY_PLUS: '+',
  KEY_MINUS: '-',

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
    return !isNaN( parseInt( KeyboardUtils.getKeyDef( domEvent ), 10 ) );
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
   * @param {KeyDef[]} keyboardUtilsKeys
   * @returns {*}
   */
  isAnyKeyEvent( domEvent, keyboardUtilsKeys ) {
    validate( domEvent, DOM_EVENT_VALIDATOR );
    return keyboardUtilsKeys.includes( KeyboardUtils.getKeyDef( domEvent ) );
  },

  /**
   * Whether or not the event was of the provided KeyboardUtils KeyDef.
   * @public
   *
   * @param {Event} domEvent
   * @param {KeyDef} keyboardUtilsKey
   * @returns {boolean}
   */
  isKeyEvent( domEvent, keyboardUtilsKey ) {
    return KeyboardUtils.getKeyDef( domEvent ) === keyboardUtilsKey;
  },

  /**
   * Returns a KeyDef that can be used to determine the keyboard keys of a KeyboardEvent object. An example
   * usage might look like
   *
   *  const key = KeyboardUtils.getKeyDef( domEvent );
   *  if ( key === KeyboardUtils.KEY_A ) {
   *    // You pressed the A key!
   *  }
   *
   * @public
   * @param {Event} domEvent
   * @returns {KeyDef|null} - null if there is no `key` property on the provided Event.
   */
  getKeyDef( domEvent ) {
    validate( domEvent, DOM_EVENT_VALIDATOR );
    return domEvent.key ? domEvent.key.toLowerCase() : null;
  },

  /**
   * Return true if the param is of type KeyDef.
   * @public
   *
   * @param {*} key
   * @returns {boolean}
   */
  isKeyDef( key ) {
    return typeof key === 'string' && key.toLowerCase() === key;
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

/**
 * @typedef KeyDef
 * @extends {String}
 * @public
 *
 * The value of KeyboardEvent.key, but lower case for easy comparison independent of modifier keys,
 * see https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/key
 *
 * In general, it is best to use globalKeyStateTracker or a similar KeyStateTracker to determine if modifier keys are
 * down.
 */

scenery.register( 'KeyboardUtils', KeyboardUtils );

export default KeyboardUtils;