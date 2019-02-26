// Copyright 2018-2019, University of Colorado Boulder

/**
 * A general type for keyboard dragging. Objects can be dragged in two dimensions with the arrow keys and with the WASD
 * keys. This can be added to a node through addInputListener for accessibility, which is mixed into Nodes with
 * the Accessibility trait.
 *
 * JavaScript does not natively handle multiple 'keydown' events at once, so we have a custom implementation that
 * tracks which keys are down and for how long in a step() function. To support keydown timing, AXON/timer is used. In
 * scenery this is supported via Display.updateOnRequestAnimationFrame(), which will step the time on each frame.
 * If using KeyboardDragListener in a more customized Display, like done in phetsims (see JOIST/Sim), the time must be
 * manually stepped (by emitting the timer).
 *
 * @author Jesse Greenberg
 * @author Michael Barlow
 */

define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var KeyboardUtil = require( 'SCENERY/accessibility/KeyboardUtil' );
  var platform = require( 'PHET_CORE/platform' );
  var scenery = require( 'SCENERY/scenery' );
  var timer = require( 'AXON/timer' );
  var Vector2 = require( 'DOT/Vector2' );

  /**
   * @constructor
   * @param {Object} options
   */
  function KeyboardDragListener( options ) {

    options = _.extend( {

      // {number|null} - While direction key is down, this will be the 1D velocity for movement. The position will
      // change this much in view coordinates every second.
      dragVelocity: 600,

      // {number|null} - If shift key down while pressing direction key, this will be the 1D delta for movement in view
      // coordinates every second.
      shiftDragVelocity: 300,

      // {Property.<Vector2>|null} - if provided, it will be synchronized with the drag location in the model
      // frame, applying provided transforms as needed. Most useful when used with transform option
      locationProperty: null,

      // {Transform3|null} - if provided, this will be the conversion between the view and model coordinate frames,
      // Usually most useful when paired with the locationProperty
      transform: null,

      // {Bounds2|null} - if provided, the model location will be constrained to be inside these bounds
      dragBounds: null,

      // {Function|null} - Called as start( event: {Event} ) when keyboard drag is started
      start: null,

      // {Function|null} - Called as drag( viewDelta: {Vector2} ) during drag
      drag: null,

      // {Function|null} - Called as end( event: {Event} ) when keyboard drag ends
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
      hotkeyInterval: 800
    }, options );

    // @private, mutable attributes declared from options, see options for info, as well as getters and setters
    this._start = options.start;
    this._drag = options.drag;
    this._end = options.end;
    this._dragBounds = options.dragBounds;
    this._transform = options.transform;
    this._locationProperty = options.locationProperty;
    this._dragVelocity = options.dragVelocity;
    this._shiftDragVelocity = options.shiftDragVelocity;
    this._downDelta = options.downDelta;
    this._shiftDownDelta = options.shiftDownDelta;
    this._moveOnHoldDelay = options.moveOnHoldDelay;
    this._moveOnHoldInterval = options.moveOnHoldInterval;
    this._hotkeyInterval = options.hotkeyInterval;

    // @private { [].{ isDown: {boolean}, timeDown: [boolean] } - tracks the state of the keyboard. JavaScript doesn't
    // handle multiple key presses, so we track which keys are currently down and update based on state of this
    // collection of objects
    // TODO: Consider a global state object for this that persists across listeners so the state of the keyboard will
    // be accurate when focus changes from one element to another, see https://github.com/phetsims/friction/issues/53
    this.keyState = [];

    // @private { [].{ keys: <Array.number>, callback: <Function> } } - groups of keys that have some behavior when
    // pressed in order. See this.addHotkeyGroup() for more information
    this.hotkeyGroups = [];

    // @private { keyCode: <number>, timeDown: <number> } - a key in a hot key group that is currently down. When all
    // keys in a group have been pressed in order, we will call the associated listener
    this.hotKeyDown = {};

    // @private {{keys: <Array.number>, callback: <Function>}|null} - the hotkey group that is currently down
    this.keyGroupDown = null;

    // @private {boolean} - when a hotkey group is pressed down, dragging will be disabled until
    // all keys are up again
    this.draggingDisabled = false;

    // @private {number} - delay before calling a keygroup listener (if keygroup is being held down), incremented in
    // step
    this.groupDownTimer = this._hotkeyInterval;

    // @private {boolean} - used to determine if the step functions updates position normally or in the 'hold to move' pattern
    this.canMove = true;

    // @private {number} - counters to allow for press-and-hold functionality that enables user to incrementally move
    // the draggable object or hold the movement key for continuous or stepped movement - values in ms
    this.moveOnHoldDelayCounter = 0;
    this.moveOnHoldIntervalCounter = 0;

    // @private {boolean} - variable to determine when the initial delay is complete
    this.delayComplete = false;

    // step the drag listener, must be removed in dispose
    var stepListener = this.step.bind( this );
    timer.addListener( stepListener );

    // @private - called in dispose
    this._disposeKeyboardDragListener = function() {
      timer.removeListener( stepListener );
    };
  }

  scenery.register( 'KeyboardDragListener', KeyboardDragListener );

  return inherit( Object, KeyboardDragListener, {


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
     * Getter for the locationProperty property, see options.locationProperty for more info.
     * @returns {Property.<Vector2>|null}
     */
    get locationProperty() { return this._locationProperty; },

    /**
     * Setter for the locationProperty property, see options.locationProperty for more info.
     * @param {Property.<Vector2>|null} locationProperty
     */
    set locationProperty( locationProperty ) { this._locationProperty = locationProperty; },

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
     * Getter for the hotkeyInterval property, see options.hotkeyInterval for more info.
     * @returns {number}
     */
    get hotkeyInterval() { return this._hotkeyInterval; },

    /**
     * Setter for the hotkeyInterval property, see options.hotkeyInterval for more info.
     * @param {number} hotkeyInterval
     */
    set hotkeyInterval( hotkeyInterval ) { this._hotkeyInterval = hotkeyInterval; },

    /**
     * Implements keyboard dragging when listener is attached to the Node, public so listener is attached
     * with addInputListener()
     *
     * @public
     * @param {Event} event
     */
    keydown: function( event ) {
      var domEvent = event.domEvent;

      // required to work with Safari and VoiceOver, otherwise arrow keys will move virtual cursor, see https://github.com/phetsims/balloons-and-static-electricity/issues/205#issuecomment-263428003
      // prevent default for WASD too, see https://github.com/phetsims/friction/issues/167
      if ( KeyboardUtil.isArrowKey( domEvent.keyCode ) || KeyboardUtil.isWASDKey( domEvent.keyCode ) ) {
        domEvent.preventDefault();
      }

      // if the key is already down, don't do anything else (we don't want to create a new keystate object
      // for a key that is already being tracked and down, nor call startDrag every keydown event)
      if ( this.keyInListDown( [ domEvent.keyCode ] ) ) { return; }

      // Prevent a VoiceOver bug where pressing multiple arrow keys at once causes the AT to send the wrong keycodes
      // through the keyup event - as a workaround, we only allow one arrow key to be down at a time. If two are pressed
      // down, we immediately clear the keystate and return
      // see https://github.com/phetsims/balloons-and-static-electricity/issues/384
      if ( platform.safari ) {
        if ( KeyboardUtil.isArrowKey( domEvent.keyCode ) ) {
          if ( this.keyInListDown( [
            KeyboardUtil.KEY_RIGHT_ARROW, KeyboardUtil.KEY_LEFT_ARROW,
            KeyboardUtil.KEY_UP_ARROW, KeyboardUtil.KEY_DOWN_ARROW ] ) ) {
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
      var positionDelta = this.shiftKeyDown() ? this._shiftDownDelta : this._downDelta;
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
     * @param {Event} event
     */
    keyup: function( event ) {
      var domEvent = event.domEvent;

      var moveKeysDown = this.movementKeysDown;

      // if the shift key is down when we navigate to the object, add it to the keystate because it won't be added until
      // the next keydown event
      if ( domEvent.keyCode === KeyboardUtil.KEY_TAB ) {
        if ( domEvent.shiftKey ) {

          // add 'shift' to the keystate until it is released again
          this.keyState.push( {
            keyDown: true,
            keyCode: KeyboardUtil.KEY_SHIFT,
            timeDown: 0 // in ms
          } );
        }
      }

      for ( var i = 0; i < this.keyState.length; i++ ) {
        if ( domEvent.keyCode === this.keyState[ i ].keyCode ) {
          this.keyState.splice( i, 1 );
        }
      }

      var moveKeysStillDown = this.movementKeysDown;
      if ( this._end ) {

        // if movement keys are no longer down after keyup, call the optional end drag function
        if ( !moveKeysStillDown && moveKeysDown !== moveKeysStillDown ) {
          this._end( event );
        }
      }

      // if keygroup keys are no longer down, clear the keygroup and reset timer
      if ( this.keyGroupDown ) {
        if ( !this.allKeysInListDown( this.keyGroupDown.keys ) ) {
          this.keyGroupDown = null;
          this.groupDownTimer = this._hotkeyInterval;
          this.draggingDisabled = false;
        }
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

      // no-op unless a key is down
      if ( this.keyState.length > 0 ) {
        // for each key that is still down, increment the tracked time that has been down
        for ( var i = 0; i < this.keyState.length; i++ ) {
          if ( this.keyState[ i ].keyDown ) {
            this.keyState[ i ].timeDown += dt;
          }
        }

        // dt is in seconds and we convert to ms
        this.moveOnHoldDelayCounter += dt * 1000;
        this.moveOnHoldIntervalCounter += dt * 1000;

        // update timer for keygroup if one is being held down
        if ( this.keyGroupDown ) {
          this.groupDownTimer += dt * 1000;
        }

        // calculate change in position from time step
        var positionVelocity = this.shiftKeyDown() ? this._shiftDragVelocity : this._dragVelocity;
        var positionDelta = dt * positionVelocity;

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
     * Handle the actual change in position of associated object based on currently pressed keys. Called in step function
     * and keydown listener.
     *
     * @param {number} delta - potential change in position in x and y for the position Property
     * @private
     */
    updatePosition: function( delta ) {

      // check to see if any hotkey combinations are down
      for ( var j = 0; j < this.hotkeyGroups.length; j++ ) {
        var hotkeysDownList = [];
        var keys = this.hotkeyGroups[ j ].keys;

        for ( var k = 0; k < keys.length; k++ ) {
          for ( var l = 0; l < this.keyState.length; l++ ) {
            if ( this.keyState[ l ].keyCode === keys[ k ] ) {
              hotkeysDownList.push( this.keyState[ l ] );
            }
          }
        }

        // the hotkeysDownList array order should match the order of the key group, so now we just need to make
        // sure that the key down times are in the right order
        var keysInOrder = false;
        for ( var m = 0; m < hotkeysDownList.length - 1; m++ ) {
          if ( hotkeysDownList[ m + 1 ] && hotkeysDownList[ m ].timeDown > hotkeysDownList[ m + 1 ].timeDown ) {
            keysInOrder = true;
          }
        }

        // if keys are in order, call the callback associated with the group, and disable dragging until
        // all hotkeys associated with that group are up again
        if ( keysInOrder ) {
          this.keyGroupDown = this.hotkeyGroups[ j ];
          if ( this.groupDownTimer >= this._hotkeyInterval ) {
            this.hotkeyGroups[ j ].callback();

            // reset timer for delay between group keygroup commands
            this.groupDownTimer = 0;
          }
        }
      }

      // if a key group is down, check to see if any of those keys are still down - if so, we will disable dragging
      // until all of them are up
      if ( this.keyGroupDown ) {
        if ( this.keyInListDown( this.keyGroupDown.keys ) ) {
          this.draggingDisabled = true;
        }
        else {
          this.draggingDisabled = false;

          // keys are no longer down, clear the group
          this.keyGroupDown = null;
        }
      }

      if ( !this.draggingDisabled ) {

        // handle the change in position
        var deltaX = 0;
        var deltaY = 0;

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
        var vectorDelta = new Vector2( deltaX, deltaY );
        if ( !vectorDelta.equals( Vector2.ZERO ) ) {

          // to model coordinates
          if ( this._transform ) {
            vectorDelta = this._transform.viewToModelDelta( vectorDelta );
          }

          // synchronize with model location
          if ( this._locationProperty ) {
            var newPosition = this._locationProperty.get().plus( vectorDelta );

            // constrain to bounds in model coordinates
            if ( this._dragBounds ) {
              newPosition = this._dragBounds.closestPointTo( newPosition );
            }

            // update the position if it is different
            if ( !newPosition.equals( this._locationProperty.get() ) ) {
              this._locationProperty.set( newPosition );
            }
          }

          // call our drag function
          if ( this._drag ) {
            this._drag( vectorDelta );
          }
        }
      }
      this.moveOnHoldIntervalCounter = 0;
      this.canMove = false;
    },

    /**
     * Returns true if any of the keys in the list are currently down.
     *
     * @param  {Array.<number>} keys
     * @returns {boolean}
     * @public
     */
    keyInListDown: function( keys ) {
      var keyIsDown = false;
      for ( var i = 0; i < this.keyState.length; i++ ) {
        if ( this.keyState[ i ].keyDown ) {
          for ( var j = 0; j < keys.length; j++ ) {
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
      var allKeysDown = true;
      for ( var i = 0; i < keys.length; i++ ) {
        for ( var j = 0; j < this.keyState.length; j++ ) {
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
      return this.keyInListDown( [ KeyboardUtil.KEY_A, KeyboardUtil.KEY_LEFT_ARROW ] );
    },

    /**
     * Returns true if the keystate indicates that a key is down that should move the object to the right.
     *
     * @public
     * @returns {boolean}
     */
    rightMovementKeysDown: function() {
      return this.keyInListDown( [ KeyboardUtil.KEY_RIGHT_ARROW, KeyboardUtil.KEY_D ] );
    },

    /**
     * Returns true if the keystate indicates that a key is down that should move the object up.
     *
     * @public
     * @returns {boolean}
     */
    upMovementKeysDown: function() {
      return this.keyInListDown( [ KeyboardUtil.KEY_UP_ARROW, KeyboardUtil.KEY_W ] );
    },

    /**
     * Returns true if the keystate indicates that a key is down that should move the upject down.
     *
     * @public
     * @returns {boolean}
     */
    downMovementKeysDown: function() {
      return this.keyInListDown( [ KeyboardUtil.KEY_DOWN_ARROW, KeyboardUtil.KEY_S ] );
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
      return this.keyInListDown( [ KeyboardUtil.KEY_ENTER ] );
    },

    /**
     * Returns true if the keystate indicates that the shift key is currently down.
     *
     * @returns {boolean}
     * @public
     */
    shiftKeyDown: function() {
      return this.keyInListDown( [ KeyboardUtil.KEY_SHIFT ] );
    },

    /**
     * Add a set of hotkeys that behave such that the desired callback will be called when
     * all keys listed in the array are pressed down in order.
     *
     * @param {Object} hotKeyGroup - { keys: [].<number>, callback: {function}, interval: {number} }
     * @public
     */
    addHotkeyGroup: function( hotKeyGroup ) {
      this.hotkeyGroups.push( hotKeyGroup );
    },

    /**
     * Add mutliple sets of hotkey groups that behave such hat the desired callback will be called
     * when all keys listed in the array are pressed down in order.  Behaves much like addHotkeyGroup,
     * but allows you to add multiple groups at one time.
     *
     * @param {[].<string>} hotKeyGroups
     * @public
     */
    addHotkeyGroups: function( hotKeyGroups ) {
      for ( var i = 0; i < hotKeyGroups.length; i++ ) {
        this.addHotkeyGroup( hotKeyGroups[ i ] );
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
     * Reset the keystate Object tracking which keys are currently pressed down.
     *
     * @public
     */
    interrupt: function() {
      this.keyState = [];
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
      return keyCode === KeyboardUtil.KEY_A || keyCode === KeyboardUtil.KEY_LEFT_ARROW;
    },

    /**
     * Returns true if the keycode corresponds to a key that should move the object to the right.
     *
     * @public
     * @returns {boolean}
     */
    isRightMovementKey: function( keyCode ) {
      return keyCode === KeyboardUtil.KEY_D || keyCode === KeyboardUtil.KEY_RIGHT_ARROW;
    },

    /**
     * Returns true if the keycode corresponds to a key that should move the object up.
     *
     * @public
     * @returns {boolean}
     */
    isUpMovementKey: function( keyCode ) {
      return keyCode === KeyboardUtil.KEY_W || keyCode === KeyboardUtil.KEY_UP_ARROW;
    },

    /**
     * Returns true if the keycode corresponds to a key that should move the object down.
     *
     * @public
     * @returns {boolean}
     */
    isDownMovementKey: function( keyCode ) {
      return keyCode === KeyboardUtil.KEY_S || keyCode === KeyboardUtil.KEY_DOWN_ARROW;
    }
  } );
} );
