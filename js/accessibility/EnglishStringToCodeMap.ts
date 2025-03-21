// Copyright 2022-2025, University of Colorado Boulder

/**
 * Maps the english key you want to use to the associated KeyboardEvent.codes for usage in listeners.
 * If a key has multiple code values, listener behavior will fire if either are pressed.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import scenery from '../scenery.js';

export type EnglishKey = keyof typeof EnglishStringToCodeMap;
export type EnglishKeyString = `${EnglishKey}`;

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
  ctrl: KeyboardUtils.CONTROL_KEYS,
  alt: KeyboardUtils.ALT_KEYS,
  shift: KeyboardUtils.SHIFT_KEYS,
  meta: KeyboardUtils.META_KEYS
};

scenery.register( 'EnglishStringToCodeMap', EnglishStringToCodeMap );
export default EnglishStringToCodeMap;

export const metaEnglishKeys: EnglishKeyString[] = [ 'ctrl', 'alt', 'shift', 'meta' ];

/**
 * Returns the first EnglishStringToCodeMap that corresponds to the provided event.code. Null if no match is found.
 * Useful when matching an english string used by KeyboardListener to the event code from a
 * SceneryEvent.domEvent.code.
 *
 * For example:
 *
 *   KeyboardUtils.eventCodeToEnglishString( 'KeyA' ) === 'a'
 *   KeyboardUtils.eventCodeToEnglishString( 'Numpad0' ) === '0'
 *   KeyboardUtils.eventCodeToEnglishString( 'Digit0' ) === '0'
 *
 * NOTE: This cannot be in KeyboardUtils because it would create a circular dependency.
 */
export const eventCodeToEnglishString = ( eventCode: string ): EnglishKeyString | null => {
  for ( const key in EnglishStringToCodeMap ) {
    if ( EnglishStringToCodeMap.hasOwnProperty( key ) &&
         ( EnglishStringToCodeMap[ key as EnglishKey ] ).includes( eventCode ) ) {
      return key as EnglishKeyString;
    }
  }
  return null;
};