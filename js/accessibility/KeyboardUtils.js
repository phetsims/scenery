// Copyright 2017-2019, University of Colorado Boulder

/**
 * Collection of utility constants and functions for managing keyboard input. Constants are keycodes. Keycode is marked
 * as deprecated, but alternatives do not have browser support. Once they do, consider replacing Event.keyCode
 * with Event.code.
 *
 * @author Jesse Greenberg
 */
define( require => {
  'use strict';

  // modules
  const scenery = require( 'SCENERY/scenery' );
  const platform = require( 'PHET_CORE/platform' );

  const KEY_RIGHT_ARROW = 39;
  const KEY_LEFT_ARROW = 37;
  const KEY_UP_ARROW = 38;
  const KEY_DOWN_ARROW = 40;
  const KEY_SHIFT = 16;
  const KEY_CTRL = 17;
  const KEY_ALT = 18;

  // constants
  var KeyboardUtils = {

    // TODO: See if these can be replaced by DOM/Browser API support
    KEY_SPACE: 32,
    KEY_ENTER: 13,
    KEY_TAB: 9,
    KEY_RIGHT_ARROW: KEY_RIGHT_ARROW,
    KEY_LEFT_ARROW: KEY_LEFT_ARROW,
    KEY_UP_ARROW: KEY_UP_ARROW,
    KEY_DOWN_ARROW: KEY_DOWN_ARROW,
    KEY_SHIFT: KEY_SHIFT,
    KEY_CTRL: KEY_CTRL,
    KEY_ALT: KEY_ALT,
    KEY_ESCAPE: 27,
    KEY_DELETE: 46,
    KEY_BACKSPACE: 8,
    KEY_PAGE_UP: 33,
    KEY_PAGE_DOWN: 34,
    KEY_END: 35,
    KEY_HOME: 36,
    KEY_PRINT_SCREEN: 44,
    KEY_0: 48,
    KEY_9: 57,
    KEY_A: 65,
    KEY_D: 68,
    KEY_C: 67,
    KEY_H: 72,
    KEY_J: 74,
    KEY_N: 78,
    KEY_S: 83,
    KEY_W: 87,

    // beware that "="" and "+" keys share the same keycode, distinguish with shfitKey Event property
    // also, these keycodes are different in Firefox, see http://www.javascripter.net/faq/keycodes.htm
    KEY_EQUALS: platform.firefox ? 61 : 187,
    KEY_PLUS: platform.firefox ? 61 : 187,
    KEY_MINUS: platform.firefox ? 173: 189,

    ARROW_KEYS: [ KEY_RIGHT_ARROW, KEY_LEFT_ARROW, KEY_RIGHT_ARROW, KEY_DOWN_ARROW ],

    // returns whether or not the keyCode corresponds to pressing an arrow key
    isArrowKey: function( keyCode ) {
      return ( keyCode === KeyboardUtils.KEY_RIGHT_ARROW || keyCode === KeyboardUtils.KEY_LEFT_ARROW ||
               keyCode === KeyboardUtils.KEY_UP_ARROW || keyCode === KeyboardUtils.KEY_DOWN_ARROW );
    },

    // returns true if keycode is one of keys used for range inputs (key codes 33 - 40, inclusive)
    isRangeKey: function( keyCode ) {
      return ( keyCode >= KeyboardUtils.KEY_PAGE_UP && keyCode <= KeyboardUtils.KEY_DOWN_ARROW );
    },

    // returns whether or not the keyCode corresponds to pressing a number key
    isNumberKey: function( keyCode ) {
      return ( keyCode > KeyboardUtils.KEY_0 && keyCode < KeyboardUtils.KEY_9 );
    },

    // returns whether or not the keyCode corresponds to one of the WASD movement keys
    isWASDKey: function( keyCode ) {
      return ( keyCode === KeyboardUtils.KEY_W || keyCode === KeyboardUtils.KEY_A ||
               keyCode === KeyboardUtils.KEY_S || keyCode === KeyboardUtils.KEY_D );
    }
  };

  scenery.register( 'KeyboardUtils', KeyboardUtils );

  return KeyboardUtils;
} );
