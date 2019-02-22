// Copyright 2018, University of Colorado Boulder

/**
 * A type that will manage the state of the keyboard. Will track which keys are being held down and for how long.
 * Offers convenience methds to determine whether or not specific keys are down like shift or enter.
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
  const timer = require( 'AXON/timer' );

  class KeyStateTracker {
    constructor() {

      // @private { Object.<number,{ keyCode: {number}, isDown: {boolean}, timeDown: [boolean] }> } - where the Object
      // keys are the keycode. JavaScript doesn't handle multiple key presses, so we track which keys are currently
      // down and update based on state of this collection of objects.
      this.keyState = {};

      const stepListener = this.step.bind( this );
      timer.addListener( stepListener );

      // @private
      this._disposeKeystateTracker = () => {
        timer.removeListener( stepListener );
      };
    }

    /**
     * Implements keyboard dragging when listener is attached to the Node, public so listener is attached
     * with addInputListener()
     *
     * Note that this event is assigned in the constructor, and not to the prototype. As of writing this,
     * `Node.addInputListener` only supports type properties as event listeners, and not the event keys as
     * prototype methods. Please see https://github.com/phetsims/scenery/issues/851 for more information.
     * @public
     * @param {Event} event
     */
    keydownUpdate( event ) {

      const domEvent = event.domEvent;

      // The dom event might have a modifier key that we weren't able to catch, if that is the case update the keystate.
      // This is likely to happen when pressing browser key commands like "ctrl + tab" to switch tabs.
      this.correctModifierKeys( domEvent );

      if ( assert && domEvent.keyCode !== KeyboardUtil.KEY_SHIFT ) {
        assert(  !!domEvent.shiftKey === !!this.shiftKeyDown, 'shift key inconsistency between event and keystate.' );
      }
      if ( assert && domEvent.keyCode !== KeyboardUtil.KEY_ALT ) {
        assert(  !!domEvent.altKey === !!this.altKeyDown, 'alt key inconsistency between event and keystate.' );
      }
      if ( assert && domEvent.keyCode !== KeyboardUtil.KEY_CTRL ) {
        assert(  !!domEvent.ctrlKey === !!this.ctrlKeyDown, 'ctrl key inconsistency between event and keystate.' );
      }

      // if the key is already down, don't do anything else (we don't want to create a new keystate object
      // for a key that is already being tracked and down)
      if ( !this.isKeyDown( domEvent.keyCode ) ) {
        this.keyState[ domEvent.keyCode ] = {
          keyDown: true,
          keyCode: domEvent.keyCode,
          timeDown: 0 // in ms
        };
      }
    }

    /**
     * Modifier keys might be part of the domEvent but the browser may or may not have received a keydown/keyup event
     * with specifically for the modifier key. This will add or remove modifier keys in that case.
     * @private
     * 
     * @param  {DOMEvent} domEvent
     */
    correctModifierKeys( domEvent ) {

      // add modifier keys if they aren't down
      if ( domEvent.shiftKey && !this.shiftKeyDown ) {
        this.keyState[ KeyboardUtil.KEY_SHIFT ] = {
          keyDown: true,
          keyCode: domEvent.keyCode,
          timeDown: 0 // in ms
        };
      }
      if ( domEvent.altKey && !this.altKeyDown ) {
        this.keyState[ KeyboardUtil.KEY_ALT ] = {
          keyDown: true,
          keyCode: domEvent.keyCode,
          timeDown: 0 // in ms
        };
      }
      if ( domEvent.ctrlKey && !this.ctrlKeyDown ) {
        this.keyState[ KeyboardUtil.KEY_CTRL ] = {
          keyDown: true,
          keyCode: domEvent.keyCode,
          timeDown: 0 // in ms
        };
      }

      // delete modifier keys if we think they are down
      if ( !domEvent.shiftKey && this.shiftKeyDown ) {
        delete this.keyState[ KeyboardUtil.KEY_SHIFT ];
      }
      if ( !domEvent.altKey && this.altKeyDown ) {
        delete this.keyState[ KeyboardUtil.KEY_ALT ];
      }
      if ( !domEvent.ctrlKey && this.ctrlKeyDown ) {
        delete this.keyState[ KeyboardUtil.KEY_CTRL ];
      }
    }

    /**
     * Behavior for keyboard 'up' DOM event. Public so it can be attached with addInputListener()
     *
     * Note that this event is assigned in the constructor, and not to the prototype. As of writing this,
     * `Node.addInputListener` only supports type properties as event listeners, and not the event keys as
     * prototype methods. Please see https://github.com/phetsims/scenery/issues/851 for more information.
     *
     * @public
     * @param {Event} event
     */
    keyupUpdate( event ) {
      const domEvent = event.domEvent;
      const keyCode = domEvent.keyCode;

      // correct keystate in case browser didn't receive keydown/keyup events for a modifier key
      this.correctModifierKeys( domEvent );

      // Remove this key data from the state - There are many cases where we might receive a keyup before keydown like
      // on first tab into scenery Display or when using specific operating system keys with the browser or PrtScn so
      // an assertion for this is too strict. See https://github.com/phetsims/scenery/issues/918
      if ( this.isKeyDown( keyCode ) ) {
        delete this.keyState[ keyCode ];
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
      const keysDown = true;
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
     * @returns {boolean} - true if the keystate indicates that the alt key is currently down.
     * @public
     */
    get altKeyDown() {
      return this.isKeyDown( KeyboardUtil.KEY_ALT );
    }

    /**
     * @returns {boolean} - true if the keystate indicates that the ctrl key is currently down.
     * @public
     */
    get ctrlKeyDown() {
      return this.isKeyDown( KeyboardUtil.KEY_CTRL );
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
        const ms = dt * 1000;

        // for each key that is still down, increment the tracked time that has been down
        for ( const i in this.keyState ) {
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
