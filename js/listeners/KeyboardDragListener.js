// Copyright 2019-2020, University of Colorado Boulder

/**
 * A general type for keyboard dragging. Objects can be dragged in two dimensions with the arrow keys and with the WASD
 * keys. This can be added to a node through addInputListener for accessibility, which is mixed into Nodes with
 * the ParallelDOM trait.
 *
 * JavaScript does not natively handle multiple 'keydown' events at once, so we have a custom implementation that
 * tracks which keys are down and for how long in a step() function. To support keydown timing, AXON/timer is used. In
 * scenery this is supported via Display.updateOnRequestAnimationFrame(), which will step the time on each frame.
 * If using KeyboardDragListener in a more customized Display, like done in phetsims (see JOIST/Sim), the time must be
 * manually stepped (by emitting the timer).
 *
 * For the purposes of this file, a "hotkey" is a collection of keys that, when pressed together in the right
 * order, fire a callback.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 * @author Michael Barlow
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import timer from '../../../axon/js/timer.js';
import Vector2 from '../../../dot/js/Vector2.js';
import inherit from '../../../phet-core/js/inherit.js';
import merge from '../../../phet-core/js/merge.js';
import platform from '../../../phet-core/js/platform.js';
import KeyboardUtils from '../accessibility/KeyboardUtils.js';
import scenery from '../scenery.js';

/**
 * @constructor
 * @param {Object} [options]
 */
function KeyboardDragListener( options ) {

  options = merge( {

    // {number|null} - While direction key is down, this will be the 1D velocity for movement. The position will
    // change this much in view coordinates every second.
    dragVelocity: 600,

    // {number|null} - If shift key down while pressing direction key, this will be the 1D delta for movement in view
    // coordinates every second.
    shiftDragVelocity: 300,

    // {Property.<Vector2>|null} - if provided, it will be synchronized with the drag position in the model
    // frame, applying provided transforms as needed. Most useful when used with transform option
    positionProperty: null,

    // {Transform3|null} - if provided, this will be the conversion between the view and model coordinate frames,
    // Usually most useful when paired with the positionProperty
    transform: null,

    // {Bounds2|null} - if provided, the model position will be constrained to be inside these bounds
    dragBounds: null,

    // {Function|null} - Called as start( event: {SceneryEvent} ) when keyboard drag is started
    start: null,

    // {Function|null} - Called as drag( viewDelta: {Vector2} ) during drag
    drag: null,

    // {Function|null} - Called as end( event: {SceneryEvent} ) when keyboard drag ends
    end: null, // called at the end of the dragging interaction

    // {number} - arrow keys must be pressed this long to begin movement set on moveOnHoldInterval, in ms
    moveOnHoldDelay: 0,

    // {number} - Time interval at which the object will change position while the arrow key is held down, in ms
    moveOnHoldInterval: 0,

    // {number} - On initial key press, how much the position Property will change in view coordinates, generally
    // only needed when there is a moveOnHoldDelay or moveOnHoldInterval. In ms.
    downDelta: 0,

    // {number} - The amount PositionProperty changes in view coordinates, generally only needed when there
    // is a moveOnHoldDelay or moveOnHoldInterval. In ms.
    shiftDownDelta: 0,

    // {number} - time interval at which holding down a hotkey group will trigger an associated listener, in ms
    hotkeyHoldInterval: 800
  }, options );

  // @private, mutable attributes declared from options, see options for info, as well as getters and setters
  this._start = options.start;
  this._drag = options.drag;
  this._end = options.end;
  this._dragBounds = options.dragBounds;
  this._transform = options.transform;
  this._positionProperty = options.positionProperty;
  this._dragVelocity = options.dragVelocity;
  this._shiftDragVelocity = options.shiftDragVelocity;
  this._downDelta = options.downDelta;
  this._shiftDownDelta = options.shiftDownDelta;
  this._moveOnHoldDelay = options.moveOnHoldDelay;
  this._moveOnHoldInterval = options.moveOnHoldInterval;
  this._hotkeyHoldInterval = options.hotkeyHoldInterval;

  // @private {Array.<{isDown:boolean, timeDown:number>} - tracks the state of the keyboard. JavaScript doesn't
  // handle multiple key presses, so we track which keys are currently down and update based on state of this
  // collection of objects. "timeDown" is in milliseconds
  // TODO: Consider a global state object for this, see https://github.com/phetsims/scenery/issues/1054
  this.keyState = [];

  // @private {Array.<{keys:<Array.number>, callback:function}>} - A list of hotkeys, each of which haing some
  // behavior when each individual key of the hotkey is  pressed in order. See this.addHotkey() for more information.
  this.hotkeys = [];

  // @private {{keys: <Array.number>, callback: <Function>}|null} - the hotkey that is currently down
  this.currentHotkey = null;

  // @private {boolean} - when a hotkey group is pressed down, dragging will be disabled until
  // any keys are up again
  this.hotkeyDisablingDragging = false;

  // @private {number} - delay before calling a keygroup listener (if keygroup is being held down), incremented in
  // step. This is initialized to the "threshold" so that the first hotkey will fire immediately. Only
  // subsequent actions while holding the hotkey should result in a delay of this much. in ms
  this.hotkeyHoldIntervalCounter = this._hotkeyHoldInterval;

  // @private {number} - counters to allow for press-and-hold functionality that enables user to incrementally move
  // the draggable object or hold the movement key for continuous or stepped movement - values in ms
  this.moveOnHoldDelayCounter = 0;
  this.moveOnHoldIntervalCounter = 0;

  // @private {boolean} - variable to determine when the initial delay is complete
  this.delayComplete = false;

  // step the drag listener, must be removed in dispose
  const stepListener = this.step.bind( this );
  timer.addListener( stepListener );

  // @private - called in dispose
  this._disposeKeyboardDragListener = function() {
    timer.removeListener( stepListener );
  };
}

scenery.register( 'KeyboardDragListener', KeyboardDragListener );

inherit( Object, KeyboardDragListener, {


  /**
   * Getter for the start property, see options.start for more info.
   * @returns {function|null}
   */
  get start() { return this._start; },

  /**
   * Setter for the start property, see options.start for more info.
   * @param {function|null} start
   */
  set start( start ) { this._start = start; },

  /**
   * Getter for the drag property, see options.drag for more info.
   * @returns {function|null}
   */
  get drag() { return this._drag; },

  /**
   * Setter for the drag property, see options.drag for more info.
   * @param {function|null} drag
   */
  set drag( drag ) { this._drag = drag; },

  /**
   * Getter for the end property, see options.end for more info.
   * @returns {function|null}
   */
  get end() { return this._end; },

  /**
   * Setter for the end property, see options.end for more info.
   * @param {function|null} end
   */
  set end( end ) { this._end = end; },

  /**
   * Getter for the dragBounds property, see options.dragBounds for more info.
   * @returns {Bounds2|null}
   */
  get dragBounds() { return this._dragBounds; },

  /**
   * Setter for the dragBounds property, see options.dragBounds for more info.
   * @param {Bounds2|null} dragBounds
   */
  set dragBounds( dragBounds ) { this._dragBounds = dragBounds; },

  /**
   * Getter for the transform property, see options.transform for more info.
   * @returns {Transform3|null}
   */
  get transform() { return this._transform; },

  /**
   * Setter for the transform property, see options.transform for more info.
   * @param {Transform3|null}transform
   */
  set transform( transform ) { this._transform = transform; },

  /**
   * Getter for the positionProperty property, see options.positionProperty for more info.
   * @returns {Property.<Vector2>|null}
   */
  get positionProperty() { return this._positionProperty; },

  /**
   * Setter for the positionProperty property, see options.positionProperty for more info.
   * @param {Property.<Vector2>|null} positionProperty
   */
  set positionProperty( positionProperty ) { this._positionProperty = positionProperty; },

  /**
   * Getter for the dragVelocity property, see options.dragVelocity for more info.
   * @returns {number|null}
   */
  get dragVelocity() { return this._dragVelocity; },

  /**
   * Setter for the dragVelocity property, see options.dragVelocity for more info.
   * @param {number|null} dragVelocity
   */
  set dragVelocity( dragVelocity ) { this._dragVelocity = dragVelocity; },

  /**
   * Getter for the shiftDragVelocity property, see options.shiftDragVelocity for more info.
   * @returns {number|null}
   */
  get shiftDragVelocity() { return this._shiftDragVelocity; },

  /**
   * Setter for the shiftDragVelocity property, see options.shiftDragVelocity for more info.
   * @param {number|null} shiftDragVelocity
   */
  set shiftDragVelocity( shiftDragVelocity ) { this._shiftDragVelocity = shiftDragVelocity; },

  /**
   * Getter for the downDelta property, see options.downDelta for more info.
   * @returns {number|null}
   */
  get downDelta() { return this._downDelta; },

  /**
   * Setter for the downDelta property, see options.downDelta for more info.
   * @param {number|null} downDelta
   */
  set downDelta( downDelta ) { this._downDelta = downDelta; },

  /**
   * Getter for the shiftDownDelta property, see options.shiftDownDelta for more info.
   * @returns {number|null}
   */
  get shiftDownDelta() { return this._shiftDownDelta; },

  /**
   * Setter for the shiftDownDelta property, see options.shiftDownDelta for more info.
   * @param {number|null} shiftDownDelta
   */
  set shiftDownDelta( shiftDownDelta ) { this._shiftDownDelta = shiftDownDelta; },

  /**
   * Getter for the moveOnHoldDelay property, see options.moveOnHoldDelay for more info.
   * @returns {number}
   */
  get moveOnHoldDelay() { return this._moveOnHoldDelay; },

  /**
   * Setter for the moveOnHoldDelay property, see options.moveOnHoldDelay for more info.
   * @param {number} moveOnHoldDelay
   */
  set moveOnHoldDelay( moveOnHoldDelay ) { this._moveOnHoldDelay = moveOnHoldDelay; },

  /**
   * Getter for the moveOnHoldInterval property, see options.moveOnHoldInterval for more info.
   * @returns {number}
   */
  get moveOnHoldInterval() { return this._moveOnHoldInterval; },

  /**
   * Setter for the moveOnHoldInterval property, see options.moveOnHoldInterval for more info.
   * @param {number} moveOnHoldInterval
   */
  set moveOnHoldInterval( moveOnHoldInterval ) { this._moveOnHoldInterval = moveOnHoldInterval; },

  /**
   * Getter for the hotkeyHoldInterval property, see options.hotkeyHoldInterval for more info.
   * @returns {number}
   */
  get hotkeyHoldInterval() { return this._hotkeyHoldInterval; },

  /**
   * Setter for the hotkeyHoldInterval property, see options.hotkeyHoldInterval for more info.
   * @param {number} hotkeyHoldInterval
   */
  set hotkeyHoldInterval( hotkeyHoldInterval ) { this._hotkeyHoldInterval = hotkeyHoldInterval; },

  /**
   * Implements keyboard dragging when listener is attached to the Node, public so listener is attached
   * with addInputListener()
   *
   * @public
   * @param {SceneryEvent} event
   */
  keydown: function( event ) {
    const domEvent = event.domEvent;

    // required to work with Safari and VoiceOver, otherwise arrow keys will move virtual cursor, see https://github.com/phetsims/balloons-and-static-electricity/issues/205#issuecomment-263428003
    // prevent default for WASD too, see https://github.com/phetsims/friction/issues/167
    if ( KeyboardUtils.isArrowKey( domEvent.keyCode ) || KeyboardUtils.isWASDKey( domEvent.keyCode ) ) {
      domEvent.preventDefault();
    }

    // reserve keyboard events for dragging to prevent default panning behavior with zoom features
    event.pointer.reserveForKeyboardDrag();

    // if the key is already down, don't do anything else (we don't want to create a new keystate object
    // for a key that is already being tracked and down, nor call startDrag every keydown event)
    if ( this.keyInListDown( [ domEvent.keyCode ] ) ) { return; }

    // Prevent a VoiceOver bug where pressing multiple arrow keys at once causes the AT to send the wrong keycodes
    // through the keyup event - as a workaround, we only allow one arrow key to be down at a time. If two are pressed
    // down, we immediately clear the keystate and return
    // see https://github.com/phetsims/balloons-and-static-electricity/issues/384
    if ( platform.safari ) {
      if ( KeyboardUtils.isArrowKey( domEvent.keyCode ) ) {
        if ( this.keyInListDown( [
          KeyboardUtils.KEY_RIGHT_ARROW, KeyboardUtils.KEY_LEFT_ARROW,
          KeyboardUtils.KEY_UP_ARROW, KeyboardUtils.KEY_DOWN_ARROW ] ) ) {
          this.interrupt();
          return;
        }
      }
    }

    // update the key state
    this.keyState.push( {
      keyDown: true,
      keyCode: domEvent.keyCode,
      timeDown: 0 // in ms
    } );

    if ( this._start ) {
      if ( this.movementKeysDown ) {
        this._start( event );
      }
    }

    // move object on first down before a delay
    const positionDelta = this.shiftKeyDown() ? this._shiftDownDelta : this._downDelta;
    this.updatePosition( positionDelta );
  },

  /**
   * Behavior for keyboard 'up' DOM event. Public so it can be attached with addInputListener()
   *
   * Note that this event is assigned in the constructor, and not to the prototype. As of writing this,
   * `Node.addInputListener` only supports type properties as event listeners, and not the event keys as
   * prototype methods. Please see https://github.com/phetsims/scenery/issues/851 for more information.
   *
   * @public
   * @param {SceneryEvent} event
   */
  keyup: function( event ) {
    const domEvent = event.domEvent;

    const moveKeysDown = this.movementKeysDown;

    // if the shift key is down when we navigate to the object, add it to the keystate because it won't be added until
    // the next keydown event
    if ( domEvent.keyCode === KeyboardUtils.KEY_TAB ) {
      if ( domEvent.shiftKey ) {

        // add 'shift' to the keystate until it is released again
        this.keyState.push( {
          keyDown: true,
          keyCode: KeyboardUtils.KEY_SHIFT,
          timeDown: 0 // in ms
        } );
      }
    }

    for ( let i = 0; i < this.keyState.length; i++ ) {
      if ( domEvent.keyCode === this.keyState[ i ].keyCode ) {
        this.keyState.splice( i, 1 );
      }
    }

    const moveKeysStillDown = this.movementKeysDown;
    if ( this._end ) {

      // if movement keys are no longer down after keyup, call the optional end drag function
      if ( !moveKeysStillDown && moveKeysDown !== moveKeysStillDown ) {
        this._end( event );
      }
    }

    // if any current hotkey keys are no longer down, clear out the current hotkey and reset.
    if ( this.currentHotkey && !this.allKeysInListDown( this.currentHotkey.keys ) ) {
      this.resetHotkeyState();
    }

    this.resetPressAndHold();
  },


  /**
   * Step function for the drag handler. JavaScript does not natively handle multiple keydown events at once,
   * so we need to track the state of the keyboard in an Object and manage dragging in this function.
   * In order for the drag handler to work.
   * @private
   *
   * @param {number} dt - in seconds
   */
  step: function( dt ) {
    const ms = dt * 1000;

    // no-op unless a key is down
    if ( this.keyState.length > 0 ) {
      // for each key that is still down, increment the tracked time that has been down
      for ( let i = 0; i < this.keyState.length; i++ ) {
        if ( this.keyState[ i ].keyDown ) {
          this.keyState[ i ].timeDown += ms;
        }
      }

      // dt is in seconds and we convert to ms
      this.moveOnHoldDelayCounter += ms;
      this.moveOnHoldIntervalCounter += ms;

      // update timer for keygroup if one is being held down
      if ( this.currentHotkey ) {
        this.hotkeyHoldIntervalCounter += ms;
      }

      // calculate change in position from time step
      const positionVelocitySeconds = this.shiftKeyDown() ? this._shiftDragVelocity : this._dragVelocity;
      const positionVelocityMilliseconds = positionVelocitySeconds / 1000;
      const positionDelta = ms * positionVelocityMilliseconds;

      if ( this.moveOnHoldDelayCounter >= this._moveOnHoldDelay && !this.delayComplete ) {
        this.updatePosition( positionDelta );
        this.delayComplete = true;
      }

      if ( this.delayComplete && this.moveOnHoldIntervalCounter >= this._moveOnHoldInterval ) {
        this.updatePosition( positionDelta );
      }
    }
  },

  /**
   * Update the state of hotkeys, and fire hotkey callbacks if one is active.
   * @private
   */
  updateHotkeys: function() {

    // check to see if any hotkey combinations are down
    for ( let j = 0; j < this.hotkeys.length; j++ ) {
      const hotkeysDownList = [];
      const keys = this.hotkeys[ j ].keys;

      for ( let k = 0; k < keys.length; k++ ) {
        for ( let l = 0; l < this.keyState.length; l++ ) {
          if ( this.keyState[ l ].keyCode === keys[ k ] ) {
            hotkeysDownList.push( this.keyState[ l ] );
          }
        }
      }

      // the hotkeysDownList array order should match the order of the key group, so now we just need to make
      // sure that the key down times are in the right order
      let keysInOrder = false;
      for ( let m = 0; m < hotkeysDownList.length - 1; m++ ) {
        if ( hotkeysDownList[ m + 1 ] && hotkeysDownList[ m ].timeDown > hotkeysDownList[ m + 1 ].timeDown ) {
          keysInOrder = true;
        }
      }

      // if keys are in order, call the callback associated with the group, and disable dragging until
      // all hotkeys associated with that group are up again
      if ( keysInOrder ) {
        this.currentHotkey = this.hotkeys[ j ];
        if ( this.hotkeyHoldIntervalCounter >= this._hotkeyHoldInterval ) {

          // Set the counter to begin counting the next interval between hotkey activations.
          this.hotkeyHoldIntervalCounter = 0;

          // call the callback last, after internal state has been updated. This solves a bug caused if this callback
          // then makes this listener interrupt.
          this.hotkeys[ j ].callback();
        }
      }
    }

    // if a key group is down, check to see if any of those keys are still down - if so, we will disable dragging
    // until all of them are up
    if ( this.currentHotkey ) {
      if ( this.keyInListDown( this.currentHotkey.keys ) ) {
        this.hotkeyDisablingDragging = true;
      }
      else {
        this.hotkeyDisablingDragging = false;

        // keys are no longer down, clear the group
        this.currentHotkey = null;
      }
    }
  },

  /**
   * Handle the actual change in position of associated object based on currently pressed keys. Called in step function
   * and keydown listener.
   *
   * @param {number} delta - potential change in position in x and y for the position Property
   * @private
   */
  updatePosition: function( delta ) {

    // hotkeys may disable dragging, so do this first
    this.updateHotkeys();

    if ( !this.hotkeyDisablingDragging ) {

      // handle the change in position
      let deltaX = 0;
      let deltaY = 0;

      if ( this.leftMovementKeysDown() ) {
        deltaX = -delta;
      }
      if ( this.rightMovementKeysDown() ) {
        deltaX = delta;
      }
      if ( this.upMovementKeysDown() ) {
        deltaY = -delta;
      }
      if ( this.downMovementKeysDown() ) {
        deltaY = delta;
      }

      // only initiate move if there was some attempted keyboard drag
      let vectorDelta = new Vector2( deltaX, deltaY );
      if ( !vectorDelta.equals( Vector2.ZERO ) ) {

        // to model coordinates
        if ( this._transform ) {
          vectorDelta = this._transform.viewToModelDelta( vectorDelta );
        }

        // synchronize with model position
        if ( this._positionProperty ) {
          let newPosition = this._positionProperty.get().plus( vectorDelta );

          // constrain to bounds in model coordinates
          if ( this._dragBounds ) {
            newPosition = this._dragBounds.closestPointTo( newPosition );
          }

          // update the position if it is different
          if ( !newPosition.equals( this._positionProperty.get() ) ) {
            this._positionProperty.set( newPosition );
          }
        }

        // call our drag function
        if ( this._drag ) {
          this._drag( vectorDelta );
        }
      }
    }
    this.moveOnHoldIntervalCounter = 0;
  },

  /**
   * Returns true if any of the keys in the list are currently down.
   *
   * @param  {Array.<number>} keys
   * @returns {boolean}
   * @public
   */
  keyInListDown: function( keys ) {
    let keyIsDown = false;
    for ( let i = 0; i < this.keyState.length; i++ ) {
      if ( this.keyState[ i ].keyDown ) {
        for ( let j = 0; j < keys.length; j++ ) {
          if ( keys[ j ] === this.keyState[ i ].keyCode ) {
            keyIsDown = true;
            break;
          }
        }
      }
      if ( keyIsDown ) {
        // no need to keep looking
        break;
      }
    }

    return keyIsDown;
  },

  /**
   * Return true if all keys in the list are currently held down.
   *
   * @param {Array.<number>} keys
   * @returns {boolean}
   * @public
   */
  allKeysInListDown: function( keys ) {
    let allKeysDown = true;
    for ( let i = 0; i < keys.length; i++ ) {
      for ( let j = 0; j < this.keyState.length; j++ ) {
        if ( this.keyState[ j ].keyDown ) {
          if ( keys[ j ] !== this.keyState ) {

            // not all keys are down, return false right away
            allKeysDown = false;
            return allKeysDown;
          }
        }
      }
    }

    // all keys must be down
    return allKeysDown;
  },

  /**
   * Returns true if the keystate indicates that a key is down that should move the object to the left.
   *
   * @public
   * @returns {boolean}
   */
  leftMovementKeysDown: function() {
    return this.keyInListDown( [ KeyboardUtils.KEY_A, KeyboardUtils.KEY_LEFT_ARROW ] );
  },

  /**
   * Returns true if the keystate indicates that a key is down that should move the object to the right.
   *
   * @public
   * @returns {boolean}
   */
  rightMovementKeysDown: function() {
    return this.keyInListDown( [ KeyboardUtils.KEY_RIGHT_ARROW, KeyboardUtils.KEY_D ] );
  },

  /**
   * Returns true if the keystate indicates that a key is down that should move the object up.
   *
   * @public
   * @returns {boolean}
   */
  upMovementKeysDown: function() {
    return this.keyInListDown( [ KeyboardUtils.KEY_UP_ARROW, KeyboardUtils.KEY_W ] );
  },

  /**
   * Returns true if the keystate indicates that a key is down that should move the upject down.
   *
   * @public
   * @returns {boolean}
   */
  downMovementKeysDown: function() {
    return this.keyInListDown( [ KeyboardUtils.KEY_DOWN_ARROW, KeyboardUtils.KEY_S ] );
  },

  /**
   * Returns true if any of the movement keys are down (arrow keys or WASD keys).
   *
   * @returns {boolean}
   * @public
   */
  getMovementKeysDown: function() {
    return this.rightMovementKeysDown() || this.leftMovementKeysDown() ||
           this.upMovementKeysDown() || this.downMovementKeysDown();
  },
  get movementKeysDown() { return this.getMovementKeysDown(); },

  /**
   * Returns true if the enter key is currently pressed down.
   *
   * @returns {boolean}
   * @public
   */
  enterKeyDown: function() {
    return this.keyInListDown( [ KeyboardUtils.KEY_ENTER ] );
  },

  /**
   * Returns true if the keystate indicates that the shift key is currently down.
   *
   * @returns {boolean}
   * @public
   */
  shiftKeyDown: function() {
    return this.keyInListDown( [ KeyboardUtils.KEY_SHIFT ] );
  },

  /**
   * Add a hotkey that behaves such that the desired callback will be called when
   * all keys listed in the array are pressed down in order.
   *
   * @param {{keys: Array.<number>, callback:function}} hotkey
   * @public
   */
  addHotkey: function( hotkey ) {
    this.hotkeys.push( hotkey );
  },

  /**
   * Add mutliple sets of hotkey groups that behave such that the desired callback will be called
   * when all keys listed in the array are pressed down in order.  Behaves much like addHotkey,
   * but allows you to add multiple groups at one time.
   *
   * For the purposes of this class, a "hotkey" is an ordered list of keys.
   *
   * @param {Array.<{keys: Array.<number>, callback:function}>} hotkeys
   * @public
   */
  addHotkeys: function( hotkeys ) {
    for ( let i = 0; i < hotkeys.length; i++ ) {
      this.addHotkey( hotkeys[ i ] );
    }
  },

  /**
   * Resets the timers and control variables for the press and hold functionality.
   *
   * @private
   */
  resetPressAndHold: function() {
    this.delayComplete = false;
    this.moveOnHoldDelayCounter = 0;
    this.moveOnHoldIntervalCounter = 0;
  },

  /**
   * Resets the timers and control variables for the hotkey functionality.
   * @private
   */
  resetHotkeyState: function() {
    this.currentHotkey = null;
    this.hotkeyHoldIntervalCounter = this._hotkeyHoldInterval; // reset to threshold so the hotkey fires immediately next time.
    this.hotkeyDisablingDragging = false;
  },

  /**
   * Reset the keystate Object tracking which keys are currently pressed down.
   *
   * @public
   */
  interrupt: function() {
    this.keyState = [];
    this.resetHotkeyState();
    this.resetPressAndHold();
  },

  /**
   * @public
   */
  dispose: function() {
    this._disposeKeyboardDragListener();
  }
}, {

  /**
   * Returns true if the keycode corresponds to a key that should move the object to the left.
   *
   * @private
   * @returns {boolean}
   */
  isLeftMovementKey: function( keyCode ) {
    return keyCode === KeyboardUtils.KEY_A || keyCode === KeyboardUtils.KEY_LEFT_ARROW;
  },

  /**
   * Returns true if the keycode corresponds to a key that should move the object to the right.
   *
   * @public
   * @returns {boolean}
   */
  isRightMovementKey: function( keyCode ) {
    return keyCode === KeyboardUtils.KEY_D || keyCode === KeyboardUtils.KEY_RIGHT_ARROW;
  },

  /**
   * Returns true if the keycode corresponds to a key that should move the object up.
   *
   * @public
   * @returns {boolean}
   */
  isUpMovementKey: function( keyCode ) {
    return keyCode === KeyboardUtils.KEY_W || keyCode === KeyboardUtils.KEY_UP_ARROW;
  },

  /**
   * Returns true if the keycode corresponds to a key that should move the object down.
   *
   * @public
   * @returns {boolean}
   */
  isDownMovementKey: function( keyCode ) {
    return keyCode === KeyboardUtils.KEY_S || keyCode === KeyboardUtils.KEY_DOWN_ARROW;
  }
} );

export default KeyboardDragListener;