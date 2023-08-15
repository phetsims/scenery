// Copyright 2023, University of Colorado Boulder

/**
 * A set of utility constants and functions for working with the english keys PhET has defined in
 * EnglishStringToCodeMap.ts.
 *
 * This is a separate file from EnglishStringToCodeMap.ts and KeyboardUtils.ts to avoid circular dependencies.
 */

import { EnglishKey } from './EnglishStringToCodeMap.js';

const ARROW_KEYS: EnglishKey[] = [ 'arrowLeft', 'arrowRight', 'arrowUp', 'arrowDown' ];
const MOVEMENT_KEYS: EnglishKey[] = [ ...ARROW_KEYS, 'w', 'a', 's', 'd' ];
const RANGE_KEYS: EnglishKey[] = [ ...ARROW_KEYS, 'pageUp', 'pageDown', 'end', 'home' ];
const NUMBER_KEYS: EnglishKey[] = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ];

const EnglishStringKeyUtils = {

  ARROW_KEYS: ARROW_KEYS,
  MOVEMENT_KEYS: MOVEMENT_KEYS,
  RANGE_KEYS: RANGE_KEYS,
  NUMBER_KEYS: NUMBER_KEYS,

  /**
   * Returns true if the key maps to an arrow key. This is an EnglishStringToCodeMap key, NOT a KeyboardEvent.code.
   */
  isArrowKey( key: EnglishKey ): boolean {
    return ARROW_KEYS.includes( key );
  },

  /**
   * Returns true if the provided key maps to a typical "movement" key, using arrow and WASD keys. This is
   * an EnglishStringToCodeMap key, NOT a KeyboardEvent.code.
   */
  isMovementKey( key: EnglishKey ): boolean {
    return MOVEMENT_KEYS.includes( key );
  },

  /**
   * Returns true if the key maps to a key used with "range" type input (like a slider). Provided key
   * should be one of EnglishStringToCodeMap's keys, NOT a KeyboardEvent.code.
   */
  isRangeKey( key: EnglishKey ): boolean {
    return RANGE_KEYS.includes( key );
  },

  /**
   * Returns true if the key is a number key. Provided key should be one of EnglishStringToCodeMap's keys, NOT a
   * KeyboardEvent.code.
   */
  isNumberKey( key: EnglishKey ): boolean {
    return NUMBER_KEYS.includes( key );
  }
};

export default EnglishStringKeyUtils;