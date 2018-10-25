// Copyright 2018, University of Colorado Boulder

/**
 * A type that will manage the state of the keyboard. Will track which keys are being held down and for how long.
 * Offers convenience methds to determine whether or not specific keys are down like shift or enter.
 *
 * This runs on phet-core's Timer class which is responsible for stepping this tracker. This is used to determine how
 * long keys have been pressed.
 *
 * @author Michael Kauzmann
 * @author Jesse Greenberg
 * @author Michael Barlow
 */
define( require => {
  'use strict';

  // modules
  const KeyboardUtil = require( 'SCENERY/accessibility/KeyboardUtil' );
  const scenery = require( 'SCENERY/scenery' );
  const timer = require( 'PHET_CORE/timer' );

  class KeyStateTracker {
    constructor() {

      // @private { Object.<number,{ keyCode: {number}, isDown: {boolean}, timeDown: [boolean] }> } - where the Object
      // keys are the keycode. JavaScript doesn't handle multiple key presses, so we track which keys are currently
      // down and update based on state of this collection of objects.
      this.keyState = {};

      // step the drag listener, must be removed in dispose
      let stepListener = this.step.bind( this );
      timer.addListener( stepListener );

      // @private
      this._disposeKeystateTracker = () => {
        timer.removeListener( stepListener );
      };
    }

    /**
     * Implements keyboard dragging when listener is attached to the Node, public so listener is attached
     * with addAccessibleInputListener()
     *
     * Note that this event is assigned in the constructor, and not to the prototype. As of writing this,
     * `Node.addAccessibleInputListener` only supports type properties as event listeners, and not the event keys as
     * prototype methods. Please see https://github.com/phetsims/scenery/issues/851 for more information.
     * @public
     * @param {DOMEvent} event
     */
    keydownUpdate( event ) {

      // required to work with Safari and VoiceOver, otherwise arrow keys will move virtual cursor
      if ( KeyboardUtil.isArrowKey( event.keyCode ) ) {
        event.preventDefault();
      }

      assert && assert( !!event.shiftKey === !!this.shiftKeyDown, 'inconsistency between event and keystate.' );

      // if the key is already down, don't do anything else (we don't want to create a new keystate object
      // for a key that is already being tracked and down)
      if ( !this.isKeyDown( event.keyCode ) ) {
        this.keyState[ event.keyCode ] = {
          keyDown: true,
          keyCode: event.keyCode,
          timeDown: 0 // in ms
        };
      }

    }

    /**
     * Behavior for keyboard 'up' DOM event. Public so it can be attached with addAccessibleInputListener()
     *
     * Note that this event is assigned in the constructor, and not to the prototype. As of writing this,
     * `Node.addAccessibleInputListener` only supports type properties as event listeners, and not the event keys as
     * prototype methods. Please see https://github.com/phetsims/scenery/issues/851 for more information.
     *
     * @public
     * @param {DOMEvent} event
     */
    keyupUpdate( event ) {

      // if the shift key is down when we navigate to the object, add it to the keystate because it won't be added until
      // the next keydown event
      if ( event.keyCode === KeyboardUtil.KEY_TAB ) {
        if ( event.shiftKey ) {

          // add 'shift' to the keystate until it is released again
          if ( !this.isKeyDown( KeyboardUtil.KEY_SHIFT ) ) {
            this.keyState[ KeyboardUtil.KEY_SHIFT ] = {
              keyDown: true,
              keyCode: KeyboardUtil.KEY_SHIFT,
              timeDown: 0 // in ms
            };
          }
        }
      }

      // remove this key data from the state
      if ( this.isKeyDown( event.keyCode ) ) {
        delete this.keyState[ event.keyCode ];
      }
    }

    /**
     * Returns true if any of the movement keys are down (arrow keys or WASD keys).
     *
     * @returns {boolean}
     * @public
     */
    get movementKeysDown() {
      return this.rightMovementKeysDown() || this.leftMovementKeysDown() ||
             this.upMovementKeysDown() || this.downMovementKeysDown();
    }

    /**
     * Returns true if a key with the keycode is currently down.
     *
     * @public
     * @param  {number} keyCode
     * @returns {boolean}
     */
    isKeyDown( keyCode ) {
      if ( !this.keyState[ keyCode ] ) {

        // key hasn't been pressed once yet
        return false;
      }

      return this.keyState[ keyCode ].keyDown;
    }

    /**
     * Returns true if any of the keys in the list are currently down.
     *
     * @param  {Array.<number>} keys - array of keycodes
     * @returns {boolean}
     * @public
     */
    isAnyKeyInListDown( keyList ) {
      for ( let i = 0; i < keyList.length; i++ ) {
        if ( this.isKeyDown( keyList[ i ] ) ) {
          return true;
        }
      }

      return false;
    }

    /**
     * Returns true if and only if all of the keys in the list are currently down.
     *
     * @param  {Array.<number>} keys - array of keycodes
     * @returns {boolean}
     * @public
     */
    areKeysDown( keyList ) {
      let keysDown = true;
      for ( let i = 0; i < keyList.length; i++ ) {
        if ( !this.isKeyDown( keyList[ i ] ) ) {
          return false;
        }
      }

      return keysDown;
    }

    /**
     * @returns {boolean} if any keys in the key state are currently down
     * @public
     */
    keysAreDown() {
      return !!Object.keys( this.keyState ).length > 0;
    }

    /**
     * @returns {boolean} - true if the enter key is currently pressed down.
     * @public
     */
    get enterKeyDown() {
      return this.isKeyDown( KeyboardUtil.KEY_ENTER );
    }

    /**
     * @returns {boolean} - true if the keystate indicates that the shift key is currently down.
     * @public
     */
    get shiftKeyDown() {
      return this.isKeyDown( KeyboardUtil.KEY_SHIFT );
    }

    /**
     * Will assert if the key isn't currently pressed down
     * @param {number} keyCode
     * @returns {number} how long the key has been down
     * @public
     */
    timeDownForKey( keyCode ) {
      assert && assert( this.isKeyDown( keyCode ), 'cannot get timeDown on a key that is not pressed down' );
      return this.keyState[ keyCode ].timeDown;
    }

    /**
     * Clear the entire state of the key tracker, basically reinitializing the instance.
     * @public
     */
    clearState() {
      this.keyState = {};
    }

    /**
     * Step function for the tracker. JavaScript does not natively handle multiple keydown events at once,
     * so we need to track the state of the keyboard in an Object and manage dragging in this function.
     * In order for the drag handler to work.
     *
     * @private
     * @param {number} dt - time in seconds that has passed since the last update
     */
    step( dt ) {

      // no-op unless a key is down
      if ( this.keysAreDown() ) {
        let ms = dt * 1000;

        // for each key that is still down, increment the tracked time that has been down
        for ( let i in this.keyState ) {
          if ( this.keyState.hasOwnProperty( i ) ) {
            if ( this.keyState[ i ].keyDown ) {
              this.keyState[ i ].timeDown += ms;
            }

          }
        }
      }
    }

    /**
     * Make eligible for garbage collection.
     * @public
     */
    dispose() {
      this._disposeKeystateTracker();
    }
  }

  return scenery.register( 'KeyStateTracker', KeyStateTracker );
} );
