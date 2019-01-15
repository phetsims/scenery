// Copyright 2017, University of Colorado Boulder

/**
 * Collection of utility constants and functions for managing keyboard input. Constants are keycodes. Keycode is marked
 * as deprecated, but alternatives do not have browser support. Once they do, consider replacing Event.keyCode
 * with Event.code.
 *
 * @author Jesse Greenberg
 */
define( function( require ) {
  'use strict';

  // modules
  var scenery = require( 'SCENERY/scenery' );

  var KEY_RIGHT_ARROW = 39;
  var KEY_LEFT_ARROW = 37;
  var KEY_UP_ARROW = 38;
  var KEY_DOWN_ARROW = 40;
  var KEY_SHIFT = 16;
  var KEY_CTRL = 17;
  var KEY_ALT = 18;

  // constants
  var KeyboardUtil = {

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
    KEY_S: 83,
    KEY_W: 87,
    KEY_A: 65,
    KEY_D: 68,
    KEY_J: 74,
    KEY_C: 67,
    KEY_N: 78,
    KEY_0: 48,
    KEY_9: 57,

    ARROW_KEYS: [ KEY_RIGHT_ARROW, KEY_LEFT_ARROW, KEY_RIGHT_ARROW, KEY_DOWN_ARROW ],

    // returns whether or not the keyCode corresponds to pressing an arrow key
    isArrowKey: function( keyCode ) {
      return ( keyCode === KeyboardUtil.KEY_RIGHT_ARROW || keyCode === KeyboardUtil.KEY_LEFT_ARROW ||
               keyCode === KeyboardUtil.KEY_UP_ARROW || keyCode === KeyboardUtil.KEY_DOWN_ARROW );
    },

    // returns true if keycode is one of keys used for range inputs (key codes 33 - 40, inclusive)
    isRangeKey: function( keyCode ) {
      return ( keyCode >= KeyboardUtil.KEY_PAGE_UP && keyCode <= KeyboardUtil.KEY_DOWN_ARROW );
    },

    // returns whether or not the keyCode corresponds to pressing a number key
    isNumberKey: function( keyCode ) {
      return ( keyCode > KeyboardUtil.KEY_0 && keyCode < KeyboardUtil.KEY_9 );
    },

    // returns whether or not the keyCode corresponds to one of the WASD movement keys
    isWASDKey: function( keyCode ) {
      return ( keyCode === KeyboardUtil.KEY_W || keyCode === KeyboardUtil.KEY_A ||
               keyCode === KeyboardUtil.KEY_S || keyCode === KeyboardUtil.KEY_D );
    }
  };

  scenery.register( 'KeyboardUtil', KeyboardUtil );

  return KeyboardUtil;
} );
