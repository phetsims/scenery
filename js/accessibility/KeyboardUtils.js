// Copyright 2017-2020, University of Colorado Boulder

/**
 * Collection of utility constants and functions for managing keyboard input. Constants are values of Event.key, as
 * well as helper functions and collections.
 *
 * @author Jesse Greenberg
 */

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

  // returns whether or not the key corresponds to pressing an arrow key
  isArrowKey( key ) {
    assert && assert( typeof key === 'string' );
    key = key.toLowerCase();
    return key === KeyboardUtils.KEY_RIGHT_ARROW ||
           key === KeyboardUtils.KEY_LEFT_ARROW ||
           key === KeyboardUtils.KEY_UP_ARROW ||
           key === KeyboardUtils.KEY_DOWN_ARROW;
  },

  // returns true if key is one of keys used for range inputs (key codes 33 - 40, inclusive)
  isRangeKey( key ) {
    assert && assert( typeof key === 'string' );
    key = key.toLowerCase();
    return key === KeyboardUtils.KEY_PAGE_UP ||
           key === KeyboardUtils.KEY_PAGE_DOWN ||
           key === KeyboardUtils.KEY_HOME ||
           key === KeyboardUtils.KEY_END ||
           KeyboardUtils.isArrowKey( key );
  },

  // returns whether or not the key corresponds to pressing a number key
  isNumberKey( key ) {
    assert && assert( typeof key === 'string' );
    key = key.toLowerCase();
    return !isNaN( parseInt( key, 10 ) );
  },

  // returns whether or not the key corresponds to one of the WASD movement keys
  isWASDKey( key ) {
    assert && assert( typeof key === 'string' );
    key = key.toLowerCase();
    return key === KeyboardUtils.KEY_W ||
           key === KeyboardUtils.KEY_A ||
           key === KeyboardUtils.KEY_S ||
           key === KeyboardUtils.KEY_D;
  },

  // returns true if the key indicates a 'movement' key in keyboard dragging
  isMovementKey( key ) {
    assert && assert( typeof key === 'string' );
    key = key.toLowerCase();
    return KeyboardUtils.MOVEMENT_KEYS.includes( key );
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