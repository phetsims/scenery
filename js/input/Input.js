// Copyright 2002-2014, University of Colorado Boulder


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
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var scenery = require( 'SCENERY/scenery' );

  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/input/Mouse' );
  require( 'SCENERY/input/Touch' );
  require( 'SCENERY/input/Pen' );
  require( 'SCENERY/input/Event' );
  require( 'SCENERY/input/Key' );
  var BatchedDOMEvent = require( 'SCENERY/input/BatchedDOMEvent' );
  var Property = require( 'AXON/Property' );

  // listenerTarget is the DOM node (window/document/element) to which DOM event listeners will be attached
  scenery.Input = function Input( rootNode, listenerTarget, batchDOMEvents, enablePointerEvents, pointFromEvent ) {
    this.rootNode = rootNode;
    this.listenerTarget = listenerTarget;
    this.batchDOMEvents = batchDOMEvents;
    this.enablePointerEvents = enablePointerEvents;
    this.pointFromEvent = pointFromEvent;
    this.displayUpdateOnEvent = false;

    //OHTWO @deprecated
    this.batchedCallbacks = []; // cleared every frame

    this.batchedEvents = [];

    //Pointer for mouse, only created lazily on first mouse event, so no mouse is allocated on tablets
    this.mouse = null;

    this.pointers = [];

    this.listenerReferences = [];

    this.eventLog = [];     // written when recording event input. can be overwritten to the empty array to reset. Strings relative to this class (prefix "rootNode.input.")
    this.logEvents = false; // can be set to true to cause Scenery to record all input calls to eventLog

    this.pointerAddedListeners = [];

    var input = this;

    // unique to this input instance
    this.onpointerdown = function onpointerdown( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, input.pointerDown, false ); };
    this.onpointerup = function onpointerup( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, input.pointerUp, true ); };
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
    this.ontouchend = function ontouchend( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, input.touchEnd, true ); };
    this.ontouchmove = function ontouchmove( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, input.touchMove, false ); };
    this.ontouchcancel = function ontouchcancel( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, input.touchCancel, false ); };
    this.onmousedown = function onmousedown( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, input.mouseDown, false ); };
    this.onmouseup = function onmouseup( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, input.mouseUp, true ); };
    this.onmousemove = function onmousemove( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, input.mouseMove, false ); };
    this.onmouseover = function onmouseover( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, input.mouseOver, false ); };
    this.onmouseout = function onmouseout( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, input.mouseOut, false ); };
    this.onkeydown = function onkeydown( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.KEY_TYPE, input.keyDown, false ); };
    this.onkeyup = function onkeyup( domEvent ) { input.batchEvent( domEvent, BatchedDOMEvent.KEY_TYPE, input.keyUp, false ); };
    this.uselessListener = function uselessListener( domEvent ) {};
  };
  var Input = scenery.Input;

  inherit( Object, Input, {
    batchEvent: function( domEvent, batchType, callback, triggerImmediate ) {
      this.batchedEvents.push( BatchedDOMEvent.createFromPool( domEvent, batchType, callback ) );
      if ( triggerImmediate || !this.batchDOMEvents ) {
        this.fireBatchedEvents();
      }
      if ( this.displayUpdateOnEvent ) {
        //OHTWO TODO: update the display
      }

      // Don't preventDefault for key events, which often need to be handled by the browser
      // (such as F5, CMD+R, CMD+OPTION+J, etc), see #332
      if ( batchType !== BatchedDOMEvent.KEY_TYPE ) {
        domEvent.preventDefault();
      }
    },

    fireBatchedEvents: function() {
      if ( this.batchedEvents.length ) {
        sceneryEventLog && sceneryEventLog( 'Input.fireBatchedEvents length:' + this.batchedEvents.length );

        // needs to be done in order
        var len = this.batchedEvents.length;
        for ( var i = 0; i < len; i++ ) {
          var batchedEvent = this.batchedEvents[i];
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
      var eventTypes = this.getUsedEventTypes();

      for ( var i = 0; i < eventTypes.length; i++ ) {
        var type = eventTypes[i];

        // work around iOS Safari 7 not sending touch events to Scenes contained in an iframe
        if ( this.listenerTarget === window ) {
          if ( addOrRemove ) {
            document.addEventListener( type, this.uselessListener );
          }
          else {
            document.removeEventListener( type, this.uselessListener );
          }
        }

        var callback = this['on' + type];
        assert && assert( !!callback );

        if ( addOrRemove ) {
          this.listenerTarget.addEventListener( type, callback, false ); // don't use capture for now
        }
        else {
          this.listenerTarget.removeEventListener( type, callback, false ); // don't use capture for now
        }
      }
    },

    addPointer: function( pointer ) {
      this.pointers.push( pointer );

      //Callback for showing pointer events.  Optimized for performance.
      if ( this.pointerAddedListeners.length ) {
        for ( var i = 0; i < this.pointerAddedListeners.length; i++ ) {
          this.pointerAddedListeners[i]( pointer );
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
        if ( this.pointers[i] === pointer ) {
          this.pointers.splice( i, 1 );
        }
      }
    },

    findTouchById: function( id ) {
      var i = this.pointers.length;
      while ( i-- ) {
        var pointer = this.pointers[i];
        if ( pointer.id === id ) {
          return pointer;
        }
      }
      return undefined;
    },

    findKeyByEvent: function( event ) {
      assert && assert( event.hasOwnProperty( 'keyCode' ) && event.hasOwnProperty( 'charCode' ), 'Assumes the KeyboardEvent has keyCode and charCode properties' );
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
      if ( this.logEvents ) { this.eventLog.push( 'mouseDown(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      var pointChanged = this.mouse.down( point, event );
      if ( pointChanged ) {
        this.moveEvent( this.mouse, event );
      }
      this.downEvent( this.mouse, event );
    },

    mouseUp: function( point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseUp(' + Input.debugText( point, event ) + ');' );
      if ( this.logEvents ) { this.eventLog.push( 'mouseUp(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      var pointChanged = this.mouse.up( point, event );
      if ( pointChanged ) {
        this.moveEvent( this.mouse, event );
      }
      this.upEvent( this.mouse, event );
    },

    mouseMove: function( point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseMove(' + Input.debugText( point, event ) + ');' );
      if ( this.logEvents ) { this.eventLog.push( 'mouseMove(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.move( point, event );
      this.moveEvent( this.mouse, event );
    },

    mouseOver: function( point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseOver(' + Input.debugText( point, event ) + ');' );
      if ( this.logEvents ) { this.eventLog.push( 'mouseOver(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.over( point, event );
      // TODO: how to handle mouse-over (and log it)
    },

    mouseOut: function( point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseOut(' + Input.debugText( point, event ) + ');' );
      if ( this.logEvents ) { this.eventLog.push( 'mouseOut(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.out( point, event );
      // TODO: how to handle mouse-out (and log it)
    },

    keyDown: function( event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'keyDown(' + Input.debugKeyEvent( event ) + ');' );
      if ( this.logEvents ) { this.eventLog.push( 'keyDown(' + Input.serializeDomEvent( event ) + ');' ); }

      var code = event.which;

      if ( pressedKeys.indexOf( code ) === -1 ) {
        pressedKeys.push( code );
      }

      // Handle TAB key (9) or 't' key temporarily for debugging
      var shiftPressed = pressedKeys.indexOf( 16 ) >= 0;
      if ( code === 9 || code === 84 ) {

        // Move the focus to the next item
        // TODO: More general focus order strategy
        var deltaIndex = shiftPressed ? -1 : +1;
        Input.moveFocus( deltaIndex );

        //TODO: Moving focus first then dispatching to focused node means newly focused node gets a fresh TAB event
        //TODO: That is probably undesirable
      }

      var key = new scenery.Key( event );
      this.addPointer( key );

      var focusedInstance = scenery.Input.focusedInstanceProperty.value;
      if ( focusedInstance ) {
        var trail = focusedInstance.node.getUniqueTrail();//TODO: Is this right?

        this.dispatchEvent( trail, 'down', key, event, true );
      }
    },

    keyUp: function( event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'keyUp(' + Input.debugKeyEvent( event ) + ');' );
      if ( this.logEvents ) { this.eventLog.push( 'keyUp(' + Input.serializeDomEvent( event ) + ');' ); }

      var code = event.which;

      // Better remove all occurences, just in case!
      while ( true ) {
        var index = pressedKeys.indexOf( code );

        if ( index > -1 ) {
          pressedKeys.splice( index, 1 );
        }
        else {
          break;
        }
      }

      var key = this.findKeyByEvent( event );
      if ( key ) {
        this.removePointer( key );
        var focusedInstance = scenery.Input.focusedInstanceProperty.value;
        if ( focusedInstance ) {
          var trail = focusedInstance.node.getUniqueTrail();//TODO: Is this right?
          this.dispatchEvent( trail, 'up', key, event, true );
        }
      }
    },

    // called for each touch point
    touchStart: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchStart(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      if ( this.logEvents ) { this.eventLog.push( 'touchStart(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var touch = new scenery.Touch( id, point, event );
      this.addPointer( touch );
      this.downEvent( touch, event );
    },

    touchEnd: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchEnd(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      if ( this.logEvents ) { this.eventLog.push( 'touchEnd(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
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
      if ( this.logEvents ) { this.eventLog.push( 'touchMove(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
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
      if ( this.logEvents ) { this.eventLog.push( 'touchCancel(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
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
      if ( this.logEvents ) { this.eventLog.push( 'penStart(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var pen = new scenery.Pen( id, point, event );
      this.addPointer( pen );
      this.downEvent( pen, event );
    },

    penEnd: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penEnd(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      if ( this.logEvents ) { this.eventLog.push( 'penEnd(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
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
      if ( this.logEvents ) { this.eventLog.push( 'penMove(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
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
      if ( this.logEvents ) { this.eventLog.push( 'penCancel(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
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
        sceneryEventLog && sceneryEventLog( 'branch change due to move event' );
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
      sceneryEventLog && sceneryEventLog( 'checking branch change: ' + trail.toString() + ' at ' + pointer.point.toString() );
      var oldTrail = pointer.trail || new scenery.Trail( this.rootNode ); // TODO: consider a static trail reference

      var lastNodeChanged = oldTrail.lastNode() !== trail.lastNode();
      if ( !lastNodeChanged && !isMove ) {
        // bail out if nothing needs to be done
        return false;
      }

      var branchIndex = scenery.Trail.branchIndex( trail, oldTrail );
      var isBranchChange = branchIndex !== trail.length || branchIndex !== oldTrail.length;
      sceneryEventLog && isBranchChange && sceneryEventLog( 'branch change from ' + oldTrail.toString() + ' to ' + trail.toString() );

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
        var pointer = this.pointers[i];
        if ( pointer.point ) {
          var changed = that.branchChangeEvents( pointer, null, false );
          if ( changed ) {
            sceneryEventLog && sceneryEventLog( 'branch change due validatePointers' );
          }
        }
      }
    },

    dispatchEvent: function( trail, type, pointer, event, bubbles ) {
      sceneryEventLog && sceneryEventLog( 'Input: ' + type + ' on ' + trail.toString() + ' for pointer ' + pointer.toString() + ' at ' + pointer.point.toString() );
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

      // TODO: better interactivity handling?
      if ( !trail.lastNode().interactive && !pointer.isKey && event && event.preventDefault ) {
        event.preventDefault();
      }
    },

    // TODO: reduce code sharing between here and dispatchToTargets!
    dispatchToPointer: function( type, pointer, inputEvent ) {
      if ( inputEvent.aborted || inputEvent.handled ) {
        return;
      }

      var specificType = pointer.type + type; // e.g. mouseup, touchup

      var pointerListeners = pointer.listeners.slice( 0 ); // defensive copy
      for ( var i = 0; i < pointerListeners.length; i++ ) {
        var listener = pointerListeners[i];

        // if a listener returns true, don't handle any more
        var aborted = false;

        if ( !aborted && listener[specificType] ) {
          listener[specificType]( inputEvent );
          aborted = inputEvent.aborted;
        }
        if ( !aborted && listener[type] ) {
          listener[type]( inputEvent );
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
        var target = trail.nodes[i];
        inputEvent.currentTarget = target;

        var listeners = target.getInputListeners();

        for ( var k = 0; k < listeners.length; k++ ) {
          var listener = listeners[k];

          // if a listener returns true, don't handle any more
          var aborted = false;

          if ( !aborted && listener[specificType] ) {
            listener[specificType]( inputEvent );
            aborted = inputEvent.aborted;
          }
          if ( !aborted && listener[type] ) {
            listener[type]( inputEvent );
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
  } );

  Input.serializeDomEvent = function serializeDomEvent( domEvent ) {
    var lines = [];
    for ( var prop in domEvent ) {
      if ( domEvent.hasOwnProperty( prop ) ) {
        // stringifying dom event object properties can cause circular references, so we avoid that completely
        if ( prop === 'touches' || prop === 'targetTouches' || prop === 'changedTouches' ) {
          var arr = [];
          for ( var i = 0; i < domEvent[prop].length; i++ ) {
            // according to spec (http://www.w3.org/TR/touch-events/), this is not an Array, but a TouchList
            var touch = domEvent[prop].item( i );

            arr.push( serializeDomEvent( touch ) );
          }
          lines.push( prop + ':[' + arr.join( ',' ) + ']' );
        }
        else {
          lines.push( prop + ':' + ( ( typeof domEvent[prop] === 'object' ) && ( domEvent[prop] !== null ) ? '{}' : JSON.stringify( domEvent[prop] ) ) );
        }
      }
    }
    return '{' + lines.join( ',' ) + '}';
  };

  Input.serializeVector2 = function( vector ) {
    return 'dot(' + vector.x + ',' + vector.y + ')';
  };

  Input.debugKeyEvent = function( domEvent ) {
    return domEvent.timeStamp + ' ' + domEvent.type;
  };

  Input.debugText = function( vector, domEvent ) {
    return vector.x + ',' + vector.y + ' ' + domEvent.timeStamp + ' ' + domEvent.type;
  };

  // maps the current MS pointer types onto the pointer spec
  Input.msPointerType = function( evt ) {
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
  };

  // Since only one element can have focus, Scenery uses a static element to track node focus.  That is, even
  // if there are multiple Displays, only one Node (across all displays) will have focus in this frame.
  Input.focusedInstanceProperty = new Property( null );

  /**
   * Adds the entire list of instances from the parent instance into the list.  List is modified, and returned.
   * This is very expensive (linear in the size of the scene graph), so use sparingly.  Currently used for focus
   * traversal.
   * @param instance
   * @param list
   * @param predicate
   */
  var flattenInstances = function( instance, list, predicate ) {
    if ( predicate( instance ) ) {
      list.push( instance );
    }
    for ( var i = 0; i < instance.children.length; i++ ) {
      flattenInstances( instance.children[i], list, predicate );
    }
    return list;
  };

  Input.focusableInstances = [];

  Input.getAllFocusableInstances = function() {
    var focusableInstances = [];
    var focusable = function( instance ) {
      return instance.node.focusable === true;
    };

    var Display = scenery.Display;//TODO: move to a traditional require statement (though may be cyclic)
    for ( var i = 0; i < Display.displays.length; i++ ) {
      var display = Display.displays[i];

      // Add to the list of all focusable items across Displays
      if ( display._baseInstance ) {
        flattenInstances( display._baseInstance, focusableInstances, focusable );
      }
    }
    return focusableInstances;
  };

  // Move the focus to the next focusable element.  Called by AccessibilityLayer.
  Input.moveFocus = function( deltaIndex ) {

    var focusableInstances = Input.focusableInstances || [];

    //If the focused instance was null, find the first focusable element.
    if ( Input.focusedInstanceProperty.value === null ) {

      Input.focusedInstanceProperty.value = focusableInstances[0];
    }
    else {
      //Find the index of the currently focused instance, and look for the next focusable instance.
      //TODO: this will fail horribly if the old node was removed, for instance.
      //TODO: Will need to be generalized, etc.

      var currentlyFocusedInstance = focusableInstances.indexOf( Input.focusedInstanceProperty.value );
      var newIndex = currentlyFocusedInstance + deltaIndex;
//      console.log( currentlyFocusedInstance, deltaIndex, newIndex, focusableInstances );

      //TODO: These loops probably not too smart here, may be better as math.
      while ( newIndex < 0 ) {
        newIndex += focusableInstances.length;
      }
      while ( newIndex >= focusableInstances.length ) {
        newIndex -= focusableInstances.length;
      }

      Input.focusedInstanceProperty.value = focusableInstances[newIndex];
    }
  };

  // Keep track of which keys are currently pressed so we know whether the shift key is down for accessibility
  // TODO: this effort is duplicated with this.pointers (which also covers different things)
  // TODO: Should they be coalesced?
  var pressedKeys = [];

  // Export some key codes for reuse in listeners.
  Input.KEY_SPACE = 32;
  Input.KEY_ENTER = 13;

  return Input;
} );
