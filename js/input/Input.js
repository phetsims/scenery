// Copyright 2013-2015, University of Colorado Boulder


/**
 * API for handling mouse / touch / keyboard events.
 *
 * A 'pointer' is an abstract way of describing either the mouse, a single touch point, or a key being pressed.
 * touch points and key presses go away after being released, whereas the mouse 'pointer' is persistent.
 *
 * Events will be called on listeners with a single event object. Supported event types are:
 * 'up', 'down', 'out', 'over', 'enter', 'exit', 'move', and 'cancel'. Scenery also supports more specific event
 * types that constrain the type of pointer, so 'mouse' + type, 'touch' + type and 'pen' + type will fire
 * on each listener before the generic event would be fined. E.g. for mouse movement, listener.mousemove will be
 * fired before listener.move.
 *
 * DOM Level 3 events spec: http://www.w3.org/TR/DOM-Level-3-Events/
 * Touch events spec: http://www.w3.org/TR/touch-events/
 * Pointer events spec draft: https://dvcs.w3.org/hg/pointerevents/raw-file/tip/pointerEvents.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var platform = require( 'PHET_CORE/platform' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Features = require( 'SCENERY/util/Features' );
  var scenery = require( 'SCENERY/scenery' );

  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/input/Mouse' );
  require( 'SCENERY/input/Touch' );
  require( 'SCENERY/input/Pen' );
  require( 'SCENERY/input/Event' );
  require( 'SCENERY/input/Key' );
  var BatchedDOMEvent = require( 'SCENERY/input/BatchedDOMEvent' );
  var Property = require( 'AXON/Property' );
  var Emitter = require( 'AXON/Emitter' );

  // Object literal makes it easy to check for the existence of an attribute (compared to [].indexOf()>=0)
  var domEventPropertiesToSerialize = {
    button: true, keyCode: true,
    deltaX: true, deltaY: true, deltaZ: true, deltaMode: true, pointerId: true,
    pointerType: true, charCode: true, which: true, clientX: true, clientY: true, changedTouches: true
  };

  /**
   * Find the index of the first occurrence of an element within an array, using equals() comparison.
   * @param array
   * @param element
   * @returns {number}
   */
  var indexOfUsingEquality = function( array, element ) {
    for ( var i = 0; i < array.length; i++ ) {
      var item = array[ i ];
      if ( item.equals( element ) ) {
        return i;
      }
    }
    return -1;
  };

  var globalDisplay = null;

  // listenerTarget is the DOM node (window/document/element) to which DOM event listeners will be attached
  function Input( display, listenerTarget, batchDOMEvents, enablePointerEvents, pointFromEvent, passiveEvents ) {
    this.display = display;
    this.rootNode = display.rootNode;
    this.listenerTarget = listenerTarget;
    this.batchDOMEvents = batchDOMEvents;
    this.enablePointerEvents = enablePointerEvents;
    this.pointFromEvent = pointFromEvent;
    this.passiveEvents = passiveEvents;
    this.displayUpdateOnEvent = false;
    globalDisplay = display;

    this.batchedEvents = [];

    //Pointer for mouse, only created lazily on first mouse event, so no mouse is allocated on tablets
    this.mouse = null;

    this.pointers = [];

    // For PhET-iO
    this.emitter = new Emitter();

    this.pointerAddedListeners = [];

    var input = this;

    // unique to this input instance
    this.onpointerdown = function onpointerdown( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, input.pointerDown, false ); scenery.Display.userGestureEmitter.emit(); };
    this.onpointerup = function onpointerup( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, input.pointerUp, true ); scenery.Display.userGestureEmitter.emit(); };
    this.onpointermove = function onpointermove( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, input.pointerMove, false ); };
    this.onpointerover = function onpointerover( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, input.pointerOver, false ); };
    this.onpointerout = function onpointerout( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, input.pointerOut, false ); };
    this.onpointercancel = function onpointercancel( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, input.pointerCancel, false ); };
    this.onMSPointerDown = function onMSPointerDown( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, input.pointerDown, false ); };
    this.onMSPointerUp = function onMSPointerUp( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, input.pointerUp, true ); };
    this.onMSPointerMove = function onMSPointerMove( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, input.pointerMove, false ); };
    this.onMSPointerOver = function onMSPointerOver( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, input.pointerOver, false ); };
    this.onMSPointerOut = function onMSPointerOut( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, input.pointerOut, false ); };
    this.onMSPointerCancel = function onMSPointerCancel( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, input.pointerCancel, false ); };
    this.ontouchstart = function ontouchstart( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, input.touchStart, false ); };
    this.ontouchend = function ontouchend( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, input.touchEnd, true ); scenery.Display.userGestureEmitter.emit(); };
    this.ontouchmove = function ontouchmove( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, input.touchMove, false ); };
    this.ontouchcancel = function ontouchcancel( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, input.touchCancel, false ); };
    this.onmousedown = function onmousedown( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, input.mouseDown, false ); scenery.Display.userGestureEmitter.emit(); };
    this.onmouseup = function onmouseup( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, input.mouseUp, true ); scenery.Display.userGestureEmitter.emit(); };
    this.onmousemove = function onmousemove( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, input.mouseMove, false ); };
    this.onmouseover = function onmouseover( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, input.mouseOver, false ); };
    this.onmouseout = function onmouseout( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, input.mouseOut, false ); };
    this.onkeydown = function onkeydown( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.KEY_TYPE, input.keyDown, false ); scenery.Display.userGestureEmitter.emit(); };
    this.onkeyup = function onkeyup( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.KEY_TYPE, input.keyUp, false ); };
    this.onwheel = function onwheel( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.WHEEL_TYPE, input.wheel, false ); };
    this.uselessListener = function uselessListener( domEvent ) {};
  }

  scenery.register( 'Input', Input );

  return inherit( Object, Input, {
      batchEvent: function( domEvent, batchType, callback, triggerImmediate ) {
        // If our display is not interactive, do not respond to any events (but still prevent default)
        if ( this.display.interactive ) {
          this.batchedEvents.push( BatchedDOMEvent.createFromPool( domEvent, batchType, callback ) );
          if ( triggerImmediate || !this.batchDOMEvents ) {
            this.fireBatchedEvents();
          }
          if ( this.displayUpdateOnEvent ) {
            //OHTWO TODO: update the display
          }
        }

        var isKey = batchType === BatchedDOMEvent.KEY_TYPE;

        // Don't preventDefault for key events, which often need to be handled by the browser
        // (such as F5, CMD+R, CMD+OPTION+J, etc), see #332
        if ( !isKey ) {
          // Always preventDefault on touch events, since we don't want mouse events triggered afterwards. See
          // http://www.html5rocks.com/en/mobile/touchandmouse/ for more information.
          // Additionally, IE had some issues with skipping prevent default, see
          // https://github.com/phetsims/scenery/issues/464 for mouse handling.
          if ( !( this.passiveEvents === true ) && ( callback !== this.mouseDown || platform.ie || platform.edge ) ) {
            domEvent.preventDefault();
          }
        }
      },

      fireBatchedEvents: function() {
        if ( this.batchedEvents.length ) {
          sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'Input.fireBatchedEvents length:' + this.batchedEvents.length );

          // needs to be done in order
          var len = this.batchedEvents.length;
          for ( var i = 0; i < len; i++ ) {
            var batchedEvent = this.batchedEvents[ i ];
            batchedEvent.run( this );
            batchedEvent.dispose();
          }
          cleanArray( this.batchedEvents );
        }
      },

      clearBatchedEvents: function() {
        this.batchedEvents.length = 0;
      },

      pointerListenerTypes: [ 'pointerdown', 'pointerup', 'pointermove', 'pointerover', 'pointerout', 'pointercancel' ],
      msPointerListenerTypes: [ 'MSPointerDown', 'MSPointerUp', 'MSPointerMove', 'MSPointerOver', 'MSPointerOut', 'MSPointerCancel' ],
      touchListenerTypes: [ 'touchstart', 'touchend', 'touchmove', 'touchcancel' ],
      mouseListenerTypes: [ 'mousedown', 'mouseup', 'mousemove', 'mouseover', 'mouseout' ],
      keyListenerTypes: [ 'keydown', 'keyup' ],
      wheelListenerTypes: [ 'wheel' ],

      // W3C spec for pointer events
      canUsePointerEvents: function() {
        return window.navigator && window.navigator.pointerEnabled && this.enablePointerEvents;
      },

      // MS spec for pointer event
      canUseMSPointerEvents: function() {
        return window.navigator && window.navigator.msPointerEnabled && this.enablePointerEvents;
      },

      getUsedEventTypes: function() {
        var eventTypes;

        if ( this.canUsePointerEvents() ) {
          // accepts pointer events corresponding to the spec at http://www.w3.org/TR/pointerevents/
          sceneryLog && sceneryLog.Input && sceneryLog.Input( 'Detected pointer events support, using that instead of mouse/touch events' );

          eventTypes = this.pointerListenerTypes;
        }
        else if ( this.canUseMSPointerEvents() ) {
          sceneryLog && sceneryLog.Input && sceneryLog.Input( 'Detected MS pointer events support, using that instead of mouse/touch events' );

          eventTypes = this.msPointerListenerTypes;
        }
        else {
          sceneryLog && sceneryLog.Input && sceneryLog.Input( 'No pointer events support detected, using mouse/touch events' );

          eventTypes = this.touchListenerTypes.concat( this.mouseListenerTypes );
        }

        eventTypes = eventTypes.concat( this.keyListenerTypes );
        eventTypes = eventTypes.concat( this.wheelListenerTypes );

        return eventTypes;
      },

      connectListeners: function() {
        this.processListeners( true );
      },

      disconnectListeners: function() {
        this.processListeners( false );
      },

      // @param addOrRemove: true if adding, false if removing
      processListeners: function( addOrRemove ) {
        var passDirectPassiveFlag = Features.passive && this.passiveEvents !== null;
        var documentOptions = passDirectPassiveFlag ? { passive: this.passiveEvents } : false;
        var mainOptions = passDirectPassiveFlag ? {
           useCapture: false,
           passive: this.passiveEvents
         } : false;
        var eventTypes = this.getUsedEventTypes();

        for ( var i = 0; i < eventTypes.length; i++ ) {
          var type = eventTypes[ i ];

          // work around iOS Safari 7 not sending touch events to Scenes contained in an iframe
          if ( this.listenerTarget === window ) {
            if ( addOrRemove ) {
              document.addEventListener( type, this.uselessListener, documentOptions );
            }
            else {
              document.removeEventListener( type, this.uselessListener, documentOptions );
            }
          }

          var callback = this[ 'on' + type ];
          assert && assert( !!callback );

          if ( addOrRemove ) {
            this.listenerTarget.addEventListener( type, callback, mainOptions ); // don't use capture for now
          }
          else {
            this.listenerTarget.removeEventListener( type, callback, mainOptions ); // don't use capture for now
          }
        }
      },

      addPointer: function( pointer ) {
        this.pointers.push( pointer );

        //Callback for showing pointer events.  Optimized for performance.
        if ( this.pointerAddedListeners.length ) {
          for ( var i = 0; i < this.pointerAddedListeners.length; i++ ) {
            this.pointerAddedListeners[ i ]( pointer );
          }
        }
      },

      addPointerAddedListener: function( listener ) {
        this.pointerAddedListeners.push( listener );
      },

      removePointerAddedListener: function( listener ) {
        var index = this.pointerAddedListeners.indexOf( listener );
        if ( index !== -1 ) {
          this.pointerAddedListeners.splice( index, index + 1 );
        }
      },

      removePointer: function( pointer ) {
        // sanity check version, will remove all instances
        for ( var i = this.pointers.length - 1; i >= 0; i-- ) {
          if ( this.pointers[ i ] === pointer ) {
            this.pointers.splice( i, 1 );
          }
        }
      },

      findTouchById: function( id ) {
        var i = this.pointers.length;
        while ( i-- ) {
          var pointer = this.pointers[ i ];
          if ( pointer.id === id ) {
            return pointer;
          }
        }
        return undefined;
      },

      findKeyByEvent: function( event ) {
        assert && assert( event.keyCode !== undefined && event.charCode !== undefined, 'Assumes the KeyboardEvent has keyCode and charCode properties' );
        var result = _.find( this.pointers, function( pointer ) {
          // TODO: also check location (if that exists), so we don't mix up left and right shift, etc.
          return pointer.event && pointer.event.keyCode === event.keyCode && pointer.event.charCode === event.charCode;
        } );
        // assert && assert( result, 'No key found for the combination of key:' + event.key + ' and location:' + event.location );
        return result;
      },

      //Init the mouse on the first mouse event (if any!)
      initMouse: function() {
        this.mouse = new scenery.Mouse();
        this.addPointer( this.mouse );
      },

      mouseDown: function( point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseDown(' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'mouseDown(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        if ( !this.mouse ) { this.initMouse(); }
        var pointChanged = this.mouse.down( point, event );
        if ( pointChanged ) {
          this.moveEvent( this.mouse, event );
        }
        this.downEvent( this.mouse, event );
      },

      mouseUp: function( point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseUp(' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'mouseUp(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        if ( !this.mouse ) { this.initMouse(); }
        var pointChanged = this.mouse.up( point, event );
        if ( pointChanged ) {
          this.moveEvent( this.mouse, event );
        }
        this.upEvent( this.mouse, event );
      },

      mouseMove: function( point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseMove(' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'mouseMove(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        if ( !this.mouse ) { this.initMouse(); }
        this.mouse.move( point, event );
        this.moveEvent( this.mouse, event );
      },

      mouseOver: function( point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseOver(' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'mouseOver(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        if ( !this.mouse ) { this.initMouse(); }
        this.mouse.over( point, event );
        // TODO: how to handle mouse-over (and log it)
      },

      mouseOut: function( point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseOut(' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'mouseOut(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        if ( !this.mouse ) { this.initMouse(); }
        this.mouse.out( point, event );
        // TODO: how to handle mouse-out (and log it)
      },

      keyDown: function( event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'keyDown(' + Input.debugKeyEvent( event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'keyDown(' + Input.serializeDomEvent( event ) + ');' ); }

        // temporary disabling
        if ( true ) { //eslint-disable-line no-constant-condition
          // if ( !this.display.options.accessibility ) {
          return;
        }

        var code = event.which;

        if ( Input.pressedKeys.indexOf( code ) === -1 ) {
          Input.pressedKeys.push( code );
        }

        // Handle TAB key
        var shiftPressed = Input.pressedKeys.indexOf( Input.KEY_SHIFT ) >= 0;

        if ( code === Input.KEY_TAB ) {

          // Move the focus to the next item
          // TODO: More general focus order strategy
          var deltaIndex = shiftPressed ? -1 : +1;
          Input.moveFocus( deltaIndex );

          //TODO: Moving focus first then dispatching to focused node means newly focused node gets a fresh TAB event
          //TODO: That is probably undesirable
        }

        var key = new scenery.Key( event );
        this.addPointer( key );

        var focusedTrail = Input.focusedTrail;
        if ( focusedTrail ) {
          this.dispatchEvent( focusedTrail, 'down', key, event, true );
        }
      },

      keyUp: function( event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'keyUp(' + Input.debugKeyEvent( event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'keyUp(' + Input.serializeDomEvent( event ) + ');' ); }

        // temporary disabling
        if ( true ) { //eslint-disable-line no-constant-condition
          // if ( !this.display.options.accessibility ) {
          return;
        }

        var code = event.which;

        // Better remove all occurences, just in case!
        while ( true ) { //eslint-disable-line no-constant-condition
          var index = Input.pressedKeys.indexOf( code );

          if ( index > -1 ) {
            Input.pressedKeys.splice( index, 1 );
          }
          else {
            break;
          }
        }

        var key = this.findKeyByEvent( event );
        if ( key ) {
          this.removePointer( key );
          var focusedTrail = Input.focusedTrail;
          if ( focusedTrail ) {
            this.dispatchEvent( focusedTrail, 'up', key, event, true );
          }
        }
      },

      // called on mouse wheels
      wheel: function( event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'wheel(' + Input.debugKeyEvent( event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'wheel(' + Input.serializeDomEvent( event ) + ');' ); }
        if ( !this.mouse ) { this.initMouse(); }
        this.mouse.wheel( event );

        // don't send mouse-wheel events if we don't yet have a mouse location!
        // TODO: Can we set the mouse location based on the wheel event?
        if ( this.mouse.point ) {
          var trail = this.rootNode.trailUnderPointer( this.mouse ) || new scenery.Trail( this.rootNode );
          this.dispatchEvent( trail, 'wheel', this.mouse, event, true );
        }
      },

      // called for each touch point
      touchStart: function( id, point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchStart(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'touchStart(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        var touch = new scenery.Touch( id, point, event );
        this.addPointer( touch );
        this.downEvent( touch, event );
      },

      touchEnd: function( id, point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchEnd(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'touchEnd(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        var touch = this.findTouchById( id );
        if ( touch ) {
          var pointChanged = touch.end( point, event );
          if ( pointChanged ) {
            this.moveEvent( touch, event );
          }
          this.removePointer( touch );
          this.upEvent( touch, event );
        }
        else {
          assert && assert( false, 'Touch not found for touchEnd: ' + id );
        }
      },

      touchMove: function( id, point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchMove(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'touchMove(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        var touch = this.findTouchById( id );
        if ( touch ) {
          touch.move( point, event );
          this.moveEvent( touch, event );
        }
        else {
          assert && assert( false, 'Touch not found for touchMove: ' + id );
        }
      },

      touchCancel: function( id, point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchCancel(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'touchCancel(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        var touch = this.findTouchById( id );
        if ( touch ) {
          var pointChanged = touch.cancel( point, event );
          if ( pointChanged ) {
            this.moveEvent( touch, event );
          }
          this.removePointer( touch );
          this.cancelEvent( touch, event );
        }
        else {
          assert && assert( false, 'Touch not found for touchCancel: ' + id );
        }
      },

      // called for each touch point
      penStart: function( id, point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penStart(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'penStart(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        var pen = new scenery.Pen( id, point, event );
        this.addPointer( pen );
        this.downEvent( pen, event );
      },

      penEnd: function( id, point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penEnd(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'penEnd(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        var pen = this.findTouchById( id );
        if ( pen ) {
          var pointChanged = pen.end( point, event );
          if ( pointChanged ) {
            this.moveEvent( pen, event );
          }
          this.removePointer( pen );
          this.upEvent( pen, event );
        }
        else {
          assert && assert( false, 'Pen not found for penEnd: ' + id );
        }
      },

      penMove: function( id, point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penMove(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'penMove(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        var pen = this.findTouchById( id );
        if ( pen ) {
          pen.move( point, event );
          this.moveEvent( pen, event );
        }
        else {
          assert && assert( false, 'Pen not found for penMove: ' + id );
        }
      },

      penCancel: function( id, point, event ) {
        sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penCancel(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
        if ( this.emitter.hasListeners() ) { this.emitter.emit1( 'penCancel(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
        var pen = this.findTouchById( id );
        if ( pen ) {
          var pointChanged = pen.cancel( point, event );
          if ( pointChanged ) {
            this.moveEvent( pen, event );
          }
          this.removePointer( pen );
          this.cancelEvent( pen, event );
        }
        else {
          assert && assert( false, 'Pen not found for penCancel: ' + id );
        }
      },

      pointerDown: function( id, type, point, event ) {
        switch( type ) {
          case 'mouse':
            // In IE for pointer down events, we want to make sure than the next interactions off the page are sent to
            // this element (it will bubble). See https://github.com/phetsims/scenery/issues/464 and
            // http://news.qooxdoo.org/mouse-capturing.
            var target = ( this.listenerTarget === window || this.listenerTarget === document ) ? document.body : this.listenerTarget;
            if ( target.setPointerCapture && event.pointerId ) {
              target.setPointerCapture( event.pointerId );
            }
            // The actual event afterwards
            this.mouseDown( point, event );
            break;
          case 'touch':
            this.touchStart( id, point, event );
            break;
          case 'pen':
            this.penStart( id, point, event );
            break;
          default:
            if ( console.log ) {
              console.log( 'Unknown pointer type: ' + type );
            }
        }
      },

      pointerUp: function( id, type, point, event ) {
        switch( type ) {
          case 'mouse':
            this.mouseUp( point, event );
            break;
          case 'touch':
            this.touchEnd( id, point, event );
            break;
          case 'pen':
            this.penEnd( id, point, event );
            break;
          default:
            if ( console.log ) {
              console.log( 'Unknown pointer type: ' + type );
            }
        }
      },

      pointerCancel: function( id, type, point, event ) {
        switch( type ) {
          case 'mouse':
            if ( console && console.log ) {
              console.log( 'WARNING: Pointer mouse cancel was received' );
            }
            break;
          case 'touch':
            this.touchCancel( id, point, event );
            break;
          case 'pen':
            this.penCancel( id, point, event );
            break;
          default:
            if ( console.log ) {
              console.log( 'Unknown pointer type: ' + type );
            }
        }
      },

      pointerMove: function( id, type, point, event ) {
        switch( type ) {
          case 'mouse':
            this.mouseMove( point, event );
            break;
          case 'touch':
            this.touchMove( id, point, event );
            break;
          case 'pen':
            this.penMove( id, point, event );
            break;
          default:
            if ( console.log ) {
              console.log( 'Unknown pointer type: ' + type );
            }
        }
      },

      pointerOver: function( id, type, point, event ) {

      },

      pointerOut: function( id, type, point, event ) {

      },

      pointerEnter: function( id, type, point, event ) {

      },

      pointerLeave: function( id, type, point, event ) {

      },

      upEvent: function( pointer, event ) {
        var trail = this.rootNode.trailUnderPointer( pointer ) || new scenery.Trail( this.rootNode );

        this.dispatchEvent( trail, 'up', pointer, event, true );

        // touch pointers are transient, so fire exit/out to the trail afterwards
        if ( pointer.isTouch ) {
          this.exitEvents( pointer, event, trail, 0, true );
        }

        pointer.trail = trail;
      },

      downEvent: function( pointer, event ) {
        var trail = this.rootNode.trailUnderPointer( pointer ) || new scenery.Trail( this.rootNode );

        // touch pointers are transient, so fire enter/over to the trail first
        if ( pointer.isTouch ) {
          this.enterEvents( pointer, event, trail, 0, true );
        }

        this.dispatchEvent( trail, 'down', pointer, event, true );

        pointer.trail = trail;
      },

      moveEvent: function( pointer, event ) {
        var changed = this.branchChangeEvents( pointer, event, true );
        if ( changed ) {
          sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'branch change due to move event' );
        }
      },

      cancelEvent: function( pointer, event ) {
        var trail = this.rootNode.trailUnderPointer( pointer ) || new scenery.Trail( this.rootNode );

        this.dispatchEvent( trail, 'cancel', pointer, event, true );

        // touch pointers are transient, so fire exit/out to the trail afterwards
        if ( pointer.isTouch ) {
          this.exitEvents( pointer, event, trail, 0, true );
        }

        pointer.trail = trail;
      },

      // return whether there was a change
      branchChangeEvents: function( pointer, event, isMove ) {
        var trail = this.rootNode.trailUnderPointer( pointer ) || new scenery.Trail( this.rootNode );
        sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
          'checking branch change: ' + trail.toString() + ' at ' + pointer.point.toString() );
        var oldTrail = pointer.trail || new scenery.Trail( this.rootNode ); // TODO: consider a static trail reference

        var lastNodeChanged = oldTrail.lastNode() !== trail.lastNode();
        if ( !lastNodeChanged && !isMove ) {
          // bail out if nothing needs to be done
          return false;
        }

        var branchIndex = scenery.Trail.branchIndex( trail, oldTrail );
        var isBranchChange = branchIndex !== trail.length || branchIndex !== oldTrail.length;
        isBranchChange && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
          'branch change from ' + oldTrail.toString() + ' to ' + trail.toString() );

        // event order matches http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
        if ( isMove ) {
          this.dispatchEvent( trail, 'move', pointer, event, true );
        }

        // we want to approximately mimic http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
        // TODO: if a node gets moved down 1 depth, it may see both an exit and enter?
        this.exitEvents( pointer, event, oldTrail, branchIndex, lastNodeChanged );
        this.enterEvents( pointer, event, trail, branchIndex, lastNodeChanged );

        pointer.trail = trail;
        return isBranchChange;
      },

      enterEvents: function( pointer, event, trail, branchIndex, lastNodeChanged ) {
        if ( trail.length > branchIndex ) {
          for ( var newIndex = trail.length - 1; newIndex >= branchIndex; newIndex-- ) {
            this.dispatchEvent( trail.slice( 0, newIndex + 1 ), 'enter', pointer, event, false );
          }
        }

        if ( lastNodeChanged ) {
          this.dispatchEvent( trail, 'over', pointer, event, true );
        }
      },

      exitEvents: function( pointer, event, trail, branchIndex, lastNodeChanged ) {
        if ( lastNodeChanged ) {
          this.dispatchEvent( trail, 'out', pointer, event, true );
        }

        if ( trail.length > branchIndex ) {
          for ( var oldIndex = branchIndex; oldIndex < trail.length; oldIndex++ ) {
            this.dispatchEvent( trail.slice( 0, oldIndex + 1 ), 'exit', pointer, event, false );
          }
        }
      },

      validatePointers: function() {
        var that = this;

        var i = this.pointers.length;
        while ( i-- ) {
          var pointer = this.pointers[ i ];
          if ( pointer.point ) {
            var changed = that.branchChangeEvents( pointer, null, false );
            if ( changed ) {
              sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'branch change due validatePointers' );
            }
          }
        }
      },

      dispatchEvent: function( trail, type, pointer, event, bubbles ) {
        sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
          'Input: ' + type + ' on ' + trail.toString() + ' for pointer ' + pointer.toString() + ' at ' + pointer.point.toString() );
        if ( !trail ) {
          try {
            throw new Error( 'falsy trail for dispatchEvent' );
          }
          catch( e ) {
            console.log( e.stack );
            throw e;
          }
        }

        // TODO: is there a way to make this event immutable?
        var inputEvent = new scenery.Event( {
          trail: trail, // {Trail} path to the leaf-most node, ordered list, from root to leaf
          type: type, // {String} what event was triggered on the listener
          pointer: pointer, // {Pointer}
          domEvent: event, // raw DOM InputEvent (TouchEvent, PointerEvent, MouseEvent,...)
          currentTarget: null, // {Node} whatever node you attached the listener to, null when passed to a Pointer,
          target: trail.lastNode() // {Node} leaf-most node in trail
        } );

        // first run through the pointer's listeners to see if one of them will handle the event
        this.dispatchToPointer( type, pointer, inputEvent );

        // if not yet handled, run through the trail in order to see if one of them will handle the event
        // at the base of the trail should be the scene node, so the scene will be notified last
        this.dispatchToTargets( trail, pointer, type, inputEvent, bubbles );
      },

      // TODO: reduce code sharing between here and dispatchToTargets!
      dispatchToPointer: function( type, pointer, inputEvent ) {
        if ( inputEvent.aborted || inputEvent.handled ) {
          return;
        }

        var specificType = pointer.type + type; // e.g. mouseup, touchup

        var pointerListeners = pointer.listeners.slice( 0 ); // defensive copy
        for ( var i = 0; i < pointerListeners.length; i++ ) {
          var listener = pointerListeners[ i ];

          // if a listener returns true, don't handle any more
          var aborted = false;

          if ( !aborted && listener[ specificType ] ) {
            listener[ specificType ]( inputEvent );
            aborted = inputEvent.aborted;
          }
          if ( pointer.firesGenericEvent && !aborted && listener[ type ] ) {
            listener[ type ]( inputEvent );
            aborted = inputEvent.aborted;
          }

          // bail out if the event is aborted, so no other listeners are triggered
          if ( aborted ) {
            return;
          }
        }
      },

      dispatchToTargets: function( trail, pointer, type, inputEvent, bubbles ) {
        if ( inputEvent.aborted || inputEvent.handled ) {
          return;
        }

        var specificType = pointer.type + type; // e.g. mouseup, touchup

        for ( var i = trail.length - 1; i >= 0; bubbles ? i-- : i = -1 ) {
          var target = trail.nodes[ i ];
          inputEvent.currentTarget = target;

          var listeners = target.getInputListeners();

          for ( var k = 0; k < listeners.length; k++ ) {
            var listener = listeners[ k ];

            // if a listener returns true, don't handle any more
            var aborted = false;

            if ( !aborted && listener[ specificType ] ) {
              listener[ specificType ]( inputEvent );
              aborted = inputEvent.aborted;
            }
            if ( pointer.firesGenericEvent && !aborted && listener[ type ] ) {
              listener[ type ]( inputEvent );
              aborted = inputEvent.aborted;
            }

            // bail out if the event is aborted, so no other listeners are triggered
            if ( aborted ) {
              return;
            }
          }

          // if the input event was handled, don't follow the trail down another level
          if ( inputEvent.handled ) {
            return;
          }
        }
      }
    },

    // Statics
    {


      serializeDomEvent: function serializeDomEvent( domEvent ) {
        var lines = [];
        for ( var prop in domEvent ) {
          if ( domEventPropertiesToSerialize[ prop ] ) {

            // stringifying dom event object properties can cause circular references, so we avoid that completely
            if ( prop === 'touches' || prop === 'targetTouches' || prop === 'changedTouches' ) {
              var arr = [];
              for ( var i = 0; i < domEvent[ prop ].length; i++ ) {

                // according to spec (http://www.w3.org/TR/touch-events/), this is not an Array, but a TouchList
                var touch = domEvent[ prop ].item( i );
                arr.push( serializeDomEvent( touch ) );
              }
              lines.push( prop + ':[' + arr.join( ',' ) + ']' );
            }
            else {
              lines.push( prop + ':' + ( ( typeof domEvent[ prop ] === 'object' ) && ( domEvent[ prop ] !== null ) ? '{}' : JSON.stringify( domEvent[ prop ] ) ) );
            }
          }
        }
        return '{' + lines.join( ',' ) + '}';
      },

      serializeVector2: function( vector ) {
        return 'new dot.Vector2(' + vector.x + ',' + vector.y + ')';
      },

      debugKeyEvent: function( domEvent ) {
        return domEvent.timeStamp + ' ' + domEvent.type;
      },

      debugText: function( vector, domEvent ) {
        return vector.x + ',' + vector.y + ' ' + domEvent.timeStamp + ' ' + domEvent.type;
      },

      // maps the current MS pointer types onto the pointer spec
      msPointerType: function( evt ) {
        if ( evt.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_TOUCH ) {
          return 'touch';
        }
        else if ( evt.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_PEN ) {
          return 'pen';
        }
        else if ( evt.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_MOUSE ) {
          return 'mouse';
        }
        else {
          return evt.pointerType; // hope for the best
        }
      },

      /*---------------------------------------------------------------------------*
       * Accessibility Support (TODO: Should this move to another file?)
       *----------------------------------------------------------------------------*/

      // Since only one element can have focus, Scenery uses a static element to track node focus.  That is, even
      // if there are multiple Displays, only one Node (across all displays) will have focus in this frame.
      focusedTrailProperty: new Property( null ),

      // ES5 getter and setter for axon-style convenience (reportedly at a performance cost)
      get focusedTrail() {
        return Input.focusedTrailProperty.get();
      },

      set focusedTrail( trail ) {
        Input.focusedTrailProperty.set( trail );
      },

      /**
       * Adds the entire list of trails from the parent instance into the list.  List is modified in-place and returned.
       * This is very expensive (linear in the size of the scene graph), so use sparingly.  Currently used for focus
       * traversal.
       * @param trail
       * @param list
       * @param predicate
       */
      flattenTrails: function( parentTrail, trail, list, predicate ) {
        while ( trail !== null ) {
          if ( predicate( trail ) && trail.isExtensionOf( parentTrail, true ) ) {
            list.push( trail );
          }
          trail = trail.next();
        }
      },

      getAllFocusableTrails: function() {
        var focusableTrails = [];
        var focusable = function( trail ) {
          return trail.isFocusable() && trail.isVisible();
        };

        // If a focus context (such as a popup) has been added, restrict the search to that instances and its children.
        if ( Input.focusContexts.length ) {
          Input.flattenTrails( Input.focusContexts[ Input.focusContexts.length - 1 ].trail, Input.focusContexts[ Input.focusContexts.length - 1 ].trail, focusableTrails, focusable );
        }
        else {
          var display = globalDisplay;

          var rootNode = display.rootNode;
          var trails = rootNode.getTrails();

          for ( var k = 0; k < trails.length; k++ ) {
            var trail = trails[ k ];

            // Add to the list of all focusable items across Displays & Trails
            Input.flattenTrails( trail, trail, focusableTrails, focusable );
          }
        }
        return focusableTrails;
      },

      getNextFocusableTrail: function( deltaIndex ) {

        // TODO: Should we persist this list across frames and do deltas for performance?
        // TODO: We used to, but it was difficult to handle instances added/removed
        // TODO: And on OSX/Chrome this seems to have good enough performance (didn't notice any qualitative slowdown)
        // TODO: Perhaps test on Mobile Safari?
        // TODO: Also, using a pattern like this could make it difficult to customize the focus traversal regions.
        var focusableTrails = Input.getAllFocusableTrails();

        //If the focused instance was null, find the first focusable element.
        if ( Input.focusedTrail === null ) {

          return focusableTrails[ 0 ];
        }
        else {
          //Find the index of the currently focused instance, and look for the next focusable instance.
          //TODO: this will fail horribly if the old node was removed, for instance.
          //TODO: Will need to be generalized, etc.

          var currentlyFocusedTrail = indexOfUsingEquality( focusableTrails, Input.focusedTrail );
          var newIndex = currentlyFocusedTrail + deltaIndex;
          //console.log( focusableInstances.length, currentlyFocusedInstance, newIndex );

          //TODO: These loops probably not too smart here, may be better as math.
          while ( newIndex < 0 ) {
            newIndex += focusableTrails.length;
          }
          while ( newIndex >= focusableTrails.length ) {
            newIndex -= focusableTrails.length;
          }

          return focusableTrails[ newIndex ];
        }
      },

      // Move the focus to the next focusable element.  Called by AccessibilityLayer.
      moveFocus: function( deltaIndex ) {
        Input.focusedTrail = Input.getNextFocusableTrail( deltaIndex );
      },

      // A focusContext is a node that focus is restricted to.  If the list is empty, then anything in the application
      // can be focused.  This is used when showing dialogs that will restrict focus.  The reason this is a stack is that
      // dialogs can spawn other dialogs.  When a dialog is dismissed, focus should return to the component that had focus
      // before the dialog was shown.
      // @private Could be a private closure var, but left public for ease of debugging.
      focusContexts: [],

      pushFocusContext: function( trail ) {
        Input.focusContexts.push( {
          trail: trail,
          previousFocusedNode: Input.focusedTrail
        } );

        // Move focus to the 1st element in the new context, but only if the focus subsystem is enabled
        // Simulation do not show focus regions unless the user has pressed tab once
        if ( Input.focusedTrail ) {
          Input.focusedTrail = Input.getAllFocusableTrails()[ 0 ];
        }
      },

      /**
       * Removes the last focus context, such as when a dialog is dismissed.  The dialog's instance is required as an argument
       * so it can be verified that it was the top element on the stack.
       */
      popFocusContext: function( trail ) {
        var top = Input.focusContexts.pop();
        assert && assert( top.trail.equals( trail ) );

        // Restore focus to the node that had focus before the popup was shown (if it still exists), but only if the
        // focus subsystem is enabled.  Simulation do not show focus regions unless the user has pressed tab once
        if ( Input.focusedTrail ) {
          Input.focusedTrail = top.previousFocusedNode;
        }
      },

      // Keep track of which keys are currently pressed so we know whether the shift key is down for accessibility
      // TODO: this effort is duplicated with this.pointers (which also covers different things)
      // TODO: Should they be coalesced?
      pressedKeys: [],

      // Export some key codes for reuse in listeners.
      // TODO: See if these can be replaced by DOM/Browser API support
      KEY_SPACE: 32,
      KEY_ENTER: 13,
      KEY_TAB: 9,
      KEY_RIGHT_ARROW: 39,
      KEY_LEFT_ARROW: 37,
      KEY_UP_ARROW: 38,
      KEY_DOWN_ARROW: 40,
      KEY_SHIFT: 16,
      KEY_ESCAPE: 27
    } );
} );
