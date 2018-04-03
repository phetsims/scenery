// Copyright 2017-2018, University of Colorado Boulder

/**
 * A general type for keyboard dragging. Objects can be dragged in two dimensions with the arrow keys and with the WASD
 * keys. This can be added to a node through addAccessibleInputListener for accessibility, which is mixed into Nodes with
 * the Accessibility trait.
 *
 * JavaScript does not natively handle multiple 'keydown' events at once, so we have a custom implementation that
 * tracks which keys are down and for how long in a step() function. This type is in scenery-phet because
 * phet-core/Timer drives the updates with step().
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
  var sceneryPhet = require( 'SCENERY_PHET/sceneryPhet' );
  var Timer = require( 'PHET_CORE/Timer' );
  var Vector2 =      require( 'DOT/Vector2' );

  /**
   * @constructor
   * @param {Object} options
   */
  function KeyboardDragListener( options ) {
    var self = this;

    options = _.extend( {

      // {number} - On initial key press, how much the position Property will change in view coordinates
      downDelta: 10,

      // {number} - On initial key press if shift is held down, amount PositionPropertychanges in view coordinates
      shiftDownDelta: 5,

      // {number|null} - while direction key is down, this will be the 1D velocity for movement, the position will
      // change this much in view coordinates every second. If null, will default to downDelta * 60 assuming 60 fps
      dragVelocity: null,

      // {number|null} - if shift key down while pressing direction key, this will be the 1D delta for movement in view
      // coordinates every second. If null, will default to shiftDownDelta * 60 assuming 60 fps
      shiftDragVelocity: null,

      // {Property.<Vector2>|null} - if provided, it will be synchronized with the drag location in the model
      // frame, applying provided transforms as needed
      locationProperty: null,

      // {Bounds2|null} - if provided, the model location will be constrained to be inside these bounds
      dragBounds: null,

      // {Transform3|null} - if provided, this will be the conversion between the view and model coordinate frames,
      // Usually most useful when paired with the locationProperty
      transform: null,

      // {Function|null} - Called as start( event: {DOMEvent} ) when keyboard drag is started
      start: null,

      // {Function|null} - Called as drag( viewDelta: {Vector2} ) during drag
      drag: null,

      // {Function|null} - Called as end( event: {DOMEvent}, viewDelta: {Vector2} ) when keyboard drag ends
      end: null, // called at the end of the dragging interaction

      // {number} - arrow keys must be pressed this long to begin movement set on interval below
      moveOnHoldDelay: 0,

      // {number} - time interval at which the object will change position while the arrow key is being held down
      moveOnHoldInterval: 0
    }, options );

    // @private, see options for more information
    this._start = options.start;
    this._drag = options.drag;
    this._end = options.end;
    this._dragBounds = options.dragBounds;
    this._transform = options.transform;
    this._locationProperty = options.locationProperty;
    this._dragVelocity = options.dragVelocity || options.downDelta * 60; // assuming same delta at 60 fps
    this._shiftDragVelocity = options.shiftDragVelocity || options.shiftDownDelta * 60; // assuming same delta at 60 fps
    this._downDelta = options.downDelta;
    this._shiftDownDelta = options.shiftDownDelta;
    this._moveOnHoldDelay = options.moveOnHoldDelay;
    this._moveOnHoldInterval = options.moveOnHoldInterval;

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

    // @private {boolean} - used to determine if the step functions updates position normally or in the 'hold to move' pattern
    this.canMove = true;

    // @private {number} - counters to allow for press-and-hold functionality that enables user to incrementally move
    // the draggable object or hold the movement key for continuous or stepped movement
    this.moveOnHoldDelayCounter = 0;
    this.moveOnHoldIntervalCounter = 0;

    // @private {boolean} - variable to determine when the initial delay is complete
    this.delayComplete = false;

    /**
     * Implements keyboard dragging when listener is attached to the node, public so listener is attached
     * with addAccessibleInputListener()
     * @public
     *
     * @param {DOMEvent} event
     */
    this.keydown = function( event ) {

      // required to work with Safari and VoiceOver, otherwise arrow keys will move virtual cursor
      if ( KeyboardUtil.isArrowKey( event.keyCode ) ) {
        event.preventDefault();
      }

      // if the key is already down, don't do anything else (we don't want to create a new keystate object
      // for a key that is already being tracked and down, nor call startDrag every keydown event)
      if ( self.keyInListDown( [ event.keyCode ] ) ) { return; }

      // Prevent a VoiceOver bug where pressing multiple arrow keys at once causes the AT to send the wrong keycodes
      // through the keyup event - as a workaround, we only allow one arrow key to be down at a time. If two are pressed
      // down, we immediately clear the keystate and return
      // see https://github.com/phetsims/balloons-and-static-electricity/issues/384
      if ( platform.safari ) {
        if ( KeyboardUtil.isArrowKey( event.keyCode ) ) {
          if ( self.keyInListDown( [ KeyboardUtil.KEY_RIGHT_ARROW, KeyboardUtil.KEY_LEFT_ARROW, KeyboardUtil.KEY_UP_ARROW, KeyboardUtil.KEY_DOWN_ARROW ] ) )  {
            self.reset();
            return;
          }
        }
      }

      // update the key state
      self.keyState.push( {
        keyDown: true,
        keyCode: event.keyCode,
        timeDown: 0 // in ms
      } );

      if ( self._start ) {
        if ( self.movementKeysDown ) {
         self._start( event );
        }
      }

      // move object on first down before a delay
      var positionDelta = self.shiftKeyDown() ? self._shiftDownDelta : self._downDelta;
      this.updatePosition( positionDelta );
    };

    /**
     * Behavior for keyboard 'up', public so it can be attached with addAccessibleInputListener()
     * @public
     *
     * @param {DOMEvent} event
     */
    this.keyup = function( event ) {
      var moveKeysDown = self.movementKeysDown;

      // if the shift key is down when we navigate to the object, add it to the keystate because it won't be added until
      // the next keydown event
      if ( event.keyCode === KeyboardUtil.KEY_TAB ) {
        if ( event.shiftKey ) {

          // add 'shift' to the keystate until it is released again
          self.keyState.push( {
            keyDown: true,
            keyCode: KeyboardUtil.KEY_SHIFT,
            timeDown: 0 // in ms
          } );
        }
      }

      for ( var i = 0; i < self.keyState.length; i++ ) {
        if ( event.keyCode === self.keyState[ i ].keyCode ) {
          self.keyState.splice( i, 1 );
        }
      }

      var moveKeysStillDown = self.movementKeysDown;
      if ( self._end ) {

        // if movement keys are no longer down after keyup, call the optional end drag function
        if ( !moveKeysStillDown && moveKeysDown !== moveKeysStillDown ) {
          self._end( event );
        }
      }

      self.resetPressAndHold();
    };

    /**
     * Behavior on blur, reset the listener when this happens, public for addAccessibleInputListener()
     * @public
     *
     * @param {DOMEvent} event
     */
    this.blur = function( event ) {
      self.reset();
    };

    // step the drag listener, must be removed in dispose
    var stepListener = this.step.bind( this );
    Timer.addStepListener( stepListener );

    // @private - called in dispose
    this._disposeKeyboardDragListener = function() {
      Timer.removeStepListener( stepListener );
    };
  }

  sceneryPhet.register( 'KeyboardDragListener', KeyboardDragListener );

  return inherit( Object, KeyboardDragListener, {

    /**
     * Step function for the drag handler. JavaScript does not natively handle multiple keydown events at once,
     * so we need to track the state of the keyboard in an Object and manage dragging in this function.
     * In order for the drag handler to work.
     *
     * @private
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
          this.hotkeyGroups[ j ].callback();
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
     * @return {boolean}
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
     * Returns true if the keystate indicates that a key is down that should move the object to the left.
     *
     * @private
     * @return {boolean}
     */
    leftMovementKeysDown: function() {
      return this.keyInListDown( [ KeyboardUtil.KEY_A, KeyboardUtil.KEY_LEFT_ARROW ] );
    },

    /**
     * Returns true if the keystate indicates that a key is down that should move the object to the right.
     *
     * @public
     * @return {boolean}
     */
    rightMovementKeysDown: function() {
      return this.keyInListDown( [ KeyboardUtil.KEY_RIGHT_ARROW, KeyboardUtil.KEY_D ] );
    },

    /**
     * Returns true if the keystate indicates that a key is down that should move the object up.
     *
     * @public
     * @return {boolean}
     */
    upMovementKeysDown: function() {
      return this.keyInListDown( [ KeyboardUtil.KEY_UP_ARROW, KeyboardUtil.KEY_W ] );
    },

    /**
     * Returns true if the keystate indicates that a key is down that should move the upject down.
     *
     * @public
     * @return {boolean}
     */
    downMovementKeysDown: function() {
      return this.keyInListDown( [ KeyboardUtil.KEY_DOWN_ARROW, KeyboardUtil.KEY_S ] );
    },

    /**
     * Returns true if any of the movement keys are down (arrow keys or WASD keys).
     *
     * @return {boolean}
     */
    getMovementKeysDown: function() {
      return this.rightMovementKeysDown() || this.leftMovementKeysDown() ||
             this.upMovementKeysDown() || this.downMovementKeysDown();
    },
    get movementKeysDown() { return this.getMovementKeysDown(); },

    /**
     * Returns true if the enter key is currently pressed down.
     *
     * @return {boolean}
     */
    enterKeyDown: function() {
      return this.keyInListDown( [ KeyboardUtil.KEY_ENTER ] );
    },

    /**
     * Returns true if the keystate indicates that the shift key is currently down.
     *
     * @return {boolean}
     */
    shiftKeyDown: function() {
      return this.keyInListDown( [ KeyboardUtil.KEY_SHIFT ] );
    },

    /**
     * Add a set of hotkeys that behave such that the desired callback will be called when
     * all keys listed in the array are pressed down in order.
     *
     * @param {Object} hotKeyGroup - { keys: [].<number>, callback: function }
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
    reset: function() {
      this.keyState = [];
      this.resetPressAndHold();
    },

    dispose: function() {
      this._disposeKeyboardDragListener();
    }
  }, {

    /**
     * Returns true if the keycode corresponds to a key that should move the object to the left.
     *
     * @private
     * @return {boolean}
     */
    isLeftMovementKey: function( keyCode ) {
      return keyCode === KeyboardUtil.KEY_A || keyCode === KeyboardUtil.KEY_LEFT_ARROW;
    },

    /**
     * Returns true if the keycode corresponds to a key that should move the object to the right.
     *
     * @public
     * @return {boolean}
     */
    isRightMovementKey: function( keyCode ) {
      return keyCode === KeyboardUtil.KEY_D || keyCode === KeyboardUtil.KEY_RIGHT_ARROW;
    },

    /**
     * Returns true if the keycode corresponds to a key that should move the object up.
     *
     * @public
     * @return {boolean}
     */
    isUpMovementKey: function( keyCode ) {
      return keyCode === KeyboardUtil.KEY_W || keyCode === KeyboardUtil.KEY_UP_ARROW;
    },

    /**
     * Returns true if the keycode corresponds to a key that should move the object down.
     *
     * @public
     * @return {boolean}
     */
    isDownMovementKey: function( keyCode ) {
      return keyCode === KeyboardUtil.KEY_S || keyCode === KeyboardUtil.KEY_DOWN_ARROW;
    },
  } );
} );
