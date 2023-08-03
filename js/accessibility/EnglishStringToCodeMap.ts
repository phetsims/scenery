// Copyright 2022-2023, University of Colorado Boulder

/**
 * Maps the english key you want to use to the associated KeyboardEvent.codes for usage in listeners.
 * If a key has multiple code values, listener behavior will fire if either are pressed.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import { KeyboardUtils, scenery } from '../imports.js';

const EnglishStringToCodeMap = {

  // Letter keys
  q: [ KeyboardUtils.KEY_Q ],
  w: [ KeyboardUtils.KEY_W ],
  e: [ KeyboardUtils.KEY_E ],
  r: [ KeyboardUtils.KEY_R ],
  t: [ KeyboardUtils.KEY_T ],
  y: [ KeyboardUtils.KEY_Y ],
  u: [ KeyboardUtils.KEY_U ],
  i: [ KeyboardUtils.KEY_I ],
  o: [ KeyboardUtils.KEY_O ],
  p: [ KeyboardUtils.KEY_P ],
  a: [ KeyboardUtils.KEY_A ],
  s: [ KeyboardUtils.KEY_S ],
  d: [ KeyboardUtils.KEY_D ],
  f: [ KeyboardUtils.KEY_F ],
  g: [ KeyboardUtils.KEY_G ],
  h: [ KeyboardUtils.KEY_H ],
  j: [ KeyboardUtils.KEY_J ],
  k: [ KeyboardUtils.KEY_K ],
  l: [ KeyboardUtils.KEY_L ],
  z: [ KeyboardUtils.KEY_Z ],
  x: [ KeyboardUtils.KEY_X ],
  c: [ KeyboardUtils.KEY_C ],
  v: [ KeyboardUtils.KEY_V ],
  b: [ KeyboardUtils.KEY_B ],
  n: [ KeyboardUtils.KEY_N ],
  m: [ KeyboardUtils.KEY_M ],

  // number keys - number and numpad
  0: [ KeyboardUtils.KEY_0, KeyboardUtils.KEY_NUMPAD_0 ],
  1: [ KeyboardUtils.KEY_1, KeyboardUtils.KEY_NUMPAD_1 ],
  2: [ KeyboardUtils.KEY_2, KeyboardUtils.KEY_NUMPAD_2 ],
  3: [ KeyboardUtils.KEY_3, KeyboardUtils.KEY_NUMPAD_3 ],
  4: [ KeyboardUtils.KEY_4, KeyboardUtils.KEY_NUMPAD_4 ],
  5: [ KeyboardUtils.KEY_5, KeyboardUtils.KEY_NUMPAD_5 ],
  6: [ KeyboardUtils.KEY_6, KeyboardUtils.KEY_NUMPAD_6 ],
  7: [ KeyboardUtils.KEY_7, KeyboardUtils.KEY_NUMPAD_7 ],
  8: [ KeyboardUtils.KEY_8, KeyboardUtils.KEY_NUMPAD_8 ],
  9: [ KeyboardUtils.KEY_9, KeyboardUtils.KEY_NUMPAD_9 ],

  // various command keys
  enter: [ KeyboardUtils.KEY_ENTER ],
  tab: [ KeyboardUtils.KEY_TAB ],
  equals: [ KeyboardUtils.KEY_EQUALS ],
  plus: [ KeyboardUtils.KEY_PLUS, KeyboardUtils.KEY_NUMPAD_PLUS ],
  minus: [ KeyboardUtils.KEY_MINUS, KeyboardUtils.KEY_NUMPAD_MINUS ],
  period: [ KeyboardUtils.KEY_PERIOD, KeyboardUtils.KEY_NUMPAD_DECIMAL ],
  escape: [ KeyboardUtils.KEY_ESCAPE ],
  delete: [ KeyboardUtils.KEY_DELETE ],
  backspace: [ KeyboardUtils.KEY_BACKSPACE ],
  pageUp: [ KeyboardUtils.KEY_PAGE_UP ],
  pageDown: [ KeyboardUtils.KEY_PAGE_DOWN ],
  end: [ KeyboardUtils.KEY_END ],
  home: [ KeyboardUtils.KEY_HOME ],
  space: [ KeyboardUtils.KEY_SPACE ],
  arrowLeft: [ KeyboardUtils.KEY_LEFT_ARROW ],
  arrowRight: [ KeyboardUtils.KEY_RIGHT_ARROW ],
  arrowUp: [ KeyboardUtils.KEY_UP_ARROW ],
  arrowDown: [ KeyboardUtils.KEY_DOWN_ARROW ],

  // modifier keys
  ctrl: [ KeyboardUtils.KEY_CONTROL_LEFT, KeyboardUtils.KEY_CONTROL_RIGHT ],
  alt: [ KeyboardUtils.KEY_ALT_LEFT, KeyboardUtils.KEY_ALT_RIGHT ],
  shift: [ KeyboardUtils.KEY_SHIFT_LEFT, KeyboardUtils.KEY_SHIFT_RIGHT ]
};

/**
 * Returns true if the key maps to an arrow key. This is an EnglishStringToCodeMap key, NOT a KeyboardEvent.code.
 */
export function isArrowKey( key: keyof typeof EnglishStringToCodeMap ): boolean {
  return key === 'arrowLeft' || key === 'arrowRight' || key === 'arrowUp' || key === 'arrowDown';
}

/**
 * Returns true if the provided key maps to a typical "movement" key, using arrow and WASD keys. This is
 * an EnglishStringToCodeMap key, NOT a KeyboardEvent.code.
 */
export function isMovementKey( key: keyof typeof EnglishStringToCodeMap ): boolean {
  return isArrowKey( key ) || key === 'w' || key === 'a' || key === 's' || key === 'd';
}

/**
 * Returns true if the key maps to a key used with "range" type input (like a slider). Provided key
 * should be one of EnglishStringToCodeMap's keys, NOT a KeyboardEvent.code.
 */
export function isRangeKey( key: keyof typeof EnglishStringToCodeMap ): boolean {
  return isArrowKey( key ) || key === 'pageUp' || key === 'pageDown' || key === 'end' || key === 'home';
}

/**
 * Returns true if the key is a number key. Provided key should be one of EnglishStringToCodeMap's keys, NOT a
 * KeyboardEvent.code.
 */
export function isNumberKey( key: keyof typeof EnglishStringToCodeMap ): boolean {
  return key === 0 || key === 1 || key === 2 || key === 3 || key === 4 || key === 5 ||
         key === 6 || key === 7 || key === 8 || key === 9;
}

scenery.register( 'EnglishStringToCodeMap', EnglishStringToCodeMap );
export default EnglishStringToCodeMap;