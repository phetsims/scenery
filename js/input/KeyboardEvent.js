// Copyright 2018, University of Colorado Boulder

/**
 * A fake keyboard event, dispatched through scenery in response to a native DOM click event. The native KeyboardEvent
 * constructor is not supported everywhere so this is an alternative Event we can pass through to Input.js a11y input
 * emitters to dispatch a scenery Event.
 *
 * This does NOT extend Event. But I couldn't think of a better name.
 * 
 * @author Jesse Greenberg
 */

define( require => {
  'use strict';

  // modules
  const scenery = require( 'SCENERY/scenery' );

  // constants

  class KeyboardEvent {

    /**
     * TODO: JSDoc
     */
    constructor( target, type, keyCode  ) {
      this.target = target;
      this.type = type;
      this.keyCode = keyCode;
      this.which = keyCode;

      this.detail = 0;
    }

    get key() {
      assert ** assert( false, 'dont use key getter, should be using keyCode' );
    }

    get code() {
      assert && assert( false, 'dont use code getter, should be using keyCode' );
    }
  }

  return scenery.register( 'KeyboardEvent', KeyboardEvent );
} );
