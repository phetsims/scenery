// Copyright 2022-2023, University of Colorado Boulder

/**
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import { KeyboardUtils, scenery } from '../imports.js';

//
const EnglishStringToCodeMap = {
  q: KeyboardUtils.KEY_Q,
  w: KeyboardUtils.KEY_W,
  e: KeyboardUtils.KEY_E,
  r: KeyboardUtils.KEY_R,
  t: KeyboardUtils.KEY_T,
  y: KeyboardUtils.KEY_Y,
  u: KeyboardUtils.KEY_U,
  i: KeyboardUtils.KEY_I,
  o: KeyboardUtils.KEY_O,
  p: KeyboardUtils.KEY_P,
  a: KeyboardUtils.KEY_A,
  s: KeyboardUtils.KEY_S,
  d: KeyboardUtils.KEY_D,
  f: KeyboardUtils.KEY_F,
  g: KeyboardUtils.KEY_G,
  h: KeyboardUtils.KEY_H,
  j: KeyboardUtils.KEY_J,
  k: KeyboardUtils.KEY_K,
  l: KeyboardUtils.KEY_L,
  z: KeyboardUtils.KEY_Z,
  x: KeyboardUtils.KEY_X,
  c: KeyboardUtils.KEY_C,
  v: KeyboardUtils.KEY_V,
  b: KeyboardUtils.KEY_B,
  n: KeyboardUtils.KEY_N,
  m: KeyboardUtils.KEY_M,
  0: KeyboardUtils.KEY_0,
  1: KeyboardUtils.KEY_1,
  2: KeyboardUtils.KEY_2,
  3: KeyboardUtils.KEY_3,
  4: KeyboardUtils.KEY_4,
  5: KeyboardUtils.KEY_5,
  6: KeyboardUtils.KEY_6,
  7: KeyboardUtils.KEY_7,
  8: KeyboardUtils.KEY_8,
  9: KeyboardUtils.KEY_9,
  [ KeyboardUtils.KEY_NUMPAD_0 ]: KeyboardUtils.KEY_NUMPAD_0,
  [ KeyboardUtils.KEY_NUMPAD_1 ]: KeyboardUtils.KEY_NUMPAD_1,
  [ KeyboardUtils.KEY_NUMPAD_2 ]: KeyboardUtils.KEY_NUMPAD_2,
  [ KeyboardUtils.KEY_NUMPAD_3 ]: KeyboardUtils.KEY_NUMPAD_3,
  [ KeyboardUtils.KEY_NUMPAD_4 ]: KeyboardUtils.KEY_NUMPAD_4,
  [ KeyboardUtils.KEY_NUMPAD_5 ]: KeyboardUtils.KEY_NUMPAD_5,
  [ KeyboardUtils.KEY_NUMPAD_6 ]: KeyboardUtils.KEY_NUMPAD_6,
  [ KeyboardUtils.KEY_NUMPAD_7 ]: KeyboardUtils.KEY_NUMPAD_7,
  [ KeyboardUtils.KEY_NUMPAD_8 ]: KeyboardUtils.KEY_NUMPAD_8,
  [ KeyboardUtils.KEY_NUMPAD_9 ]: KeyboardUtils.KEY_NUMPAD_9,
  [ KeyboardUtils.KEY_NUMPAD_DECIMAL ]: KeyboardUtils.KEY_NUMPAD_DECIMAL,
  [ KeyboardUtils.KEY_NUMPAD_DECIMAL ]: KeyboardUtils.KEY_NUMPAD_DECIMAL,
  [ KeyboardUtils.KEY_NUMPAD_PLUS ]: KeyboardUtils.KEY_NUMPAD_PLUS,
  [ KeyboardUtils.KEY_NUMPAD_MINUS ]: KeyboardUtils.KEY_NUMPAD_MINUS,

  ctrl: KeyboardUtils.KEY_CONTROL,
  alt: KeyboardUtils.KEY_ALT,
  shift: KeyboardUtils.KEY_SHIFT,
  ctrlLeft: KeyboardUtils.KEY_CONTROL_LEFT,
  ctrlRight: KeyboardUtils.KEY_CONTROL_RIGHT,
  shiftLeft: KeyboardUtils.KEY_SHIFT_LEFT,
  shiftRight: KeyboardUtils.KEY_SHIFT_RIGHT,
  altLeft: KeyboardUtils.KEY_ALT_LEFT,
  altRight: KeyboardUtils.KEY_ALT_RIGHT,

  enter: KeyboardUtils.KEY_ENTER,
  tab: KeyboardUtils.KEY_TAB,
  equals: KeyboardUtils.KEY_EQUALS,
  plus: KeyboardUtils.KEY_PLUS,
  minus: KeyboardUtils.KEY_MINUS,
  period: KeyboardUtils.KEY_PERIOD,
  escape: KeyboardUtils.KEY_ESCAPE,
  delete: KeyboardUtils.KEY_DELETE,
  backspace: KeyboardUtils.KEY_BACKSPACE,
  page_up: KeyboardUtils.KEY_PAGE_UP,
  page_down: KeyboardUtils.KEY_PAGE_DOWN,
  end: KeyboardUtils.KEY_END,
  home: KeyboardUtils.KEY_HOME,

  space: KeyboardUtils.KEY_SPACE,
  arrowLeft: KeyboardUtils.KEY_LEFT_ARROW,
  arrowRight: KeyboardUtils.KEY_RIGHT_ARROW,
  arrowUp: KeyboardUtils.KEY_UP_ARROW,
  arrowDown: KeyboardUtils.KEY_DOWN_ARROW
};

scenery.register( 'EnglishStringToCodeMap', EnglishStringToCodeMap );
export default EnglishStringToCodeMap;