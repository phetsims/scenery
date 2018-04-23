// Copyright 2013-2016, University of Colorado Boulder

/**
 * Main handler for user-input events in Scenery.
 *
 * *** Adding input handling to a display
 *
 * Displays do not have event listeners attached by default. To initialize the event system (that will set up
 * listeners), use one of Display's initialize*Events functions.
 *
 * *** Pointers
 *
 * A 'pointer' is an abstract way of describing a mouse, a single touch point, or a pen/stylus, similar to in the
 * Pointer Events specification (https://dvcs.w3.org/hg/pointerevents/raw-file/tip/pointerEvents.html). Touch and pen
 * pointers are transient, created when the relevant DOM down event occurs and released when corresponding the DOM up
 * or cancel event occurs. However, the mouse pointer is persistent.
 *
 * Input event listeners can be added to {Node}s directly, or to a pointer. When a DOM event is received, it is first
 * broken up into multiple events (if necessary, e.g. multiple touch points), then the dispatch is handled for each
 * individual Scenery event. Events are first fired for any listeners attached to the pointer that caused the event,
 * then fire on the node directly under the pointer, and if applicable, bubble up the graph to the Scene from which the
 * event was triggered. Events are not fired directly on nodes that are not under the pointer at the time of the event.
 * To handle many common patterns (like button presses, where mouse-ups could happen when not over the button), it is
 * necessary to add those move/up listeners to the pointer itself.
 *
 * *** Listeners and Events
 *
 * Event listeners are added with node.addInputListener( listener ) and pointer.addInputListener( listener ).
 * This listener can be an arbitrary object, and the listener will be triggered by calling listener[eventType]( event ),
 * where eventType is one of the event types as described below, and event is a Scenery event with the
 * following properties:
 * - trail {Trail} - Points to the node under the pointer
 * - pointer {Pointer} - The pointer that triggered the event. Additional information about the mouse/touch/pen can be
 *                       obtained from the pointer, for example event.pointer.point.
 * - type {string} - The base type of the event (e.g. for touch down events, it will always just be "down").
 * - domEvent {Event} - The underlying DOM event that triggered this Scenery event. The DOM event may correspond to
 *                      multiple Scenery events, particularly for touch events. This could be a TouchEvent,
 *                      PointerEvent, MouseEvent, MSPointerEvent, etc.
 * - target {Node} - The leaf-most Node in the trail.
 * - currentTarget {Node} - The Node to which the listener being fired is attached, or null if the listener is being
 *                          fired directly from a pointer.
 *
 * *** Event Types
 *
 * Scenery will fire the following base event types:
 *
 * - down: Triggered when a pointer is pressed down. Touch / pen pointers are created for each down event, and are
 *         active until an up/cancel event is sent.
 * - up: Triggered when a pointer is released normally. Touch / pen pointers will not have any more events associated
 *       with them after an up event.
 * - cancel: Triggered when a pointer is canceled abnormally. Touch / pen pointers will not have any more events
 *           associated with them after an up event.
 * - move: Triggered when a pointer moves.
 * - wheel: Triggered when the (mouse) wheel is scrolled. The associated pointer will have wheelDelta information.
 * - enter: Triggered when a pointer moves over a Node or one of its children. Does not bubble up. Mirrors behavior from
 *          the DOM mouseenter (http://www.w3.org/TR/DOM-Level-3-Events/#event-type-mouseenter)
 * - exit:  Triggered when a pointer moves out from over a Node or one of its children. Does not bubble up. Mirrors
 *          behavior from the DOM mouseleave (http://www.w3.org/TR/DOM-Level-3-Events/#event-type-mouseleave).
 * - over: Triggered when a pointer moves over a Node (not including its children). Mirrors behavior from the DOM
 *         mouseover (http://www.w3.org/TR/DOM-Level-3-Events/#event-type-mouseover).
 * - out: Triggered when a pointer moves out from over a Node (not including its children). Mirrors behavior from the
 *        DOM mouseout (http://www.w3.org/TR/DOM-Level-3-Events/#event-type-mouseout).
 *
 * Before firing the base event type (for example, 'move'), Scenery will also fire an event specific to the type of
 * pointer. For mice, it will fire 'mousemove', for touch events it will fire 'touchmove', and for pen events it will
 * fire 'penmove'. Similarly, for any type of event, it will first fire pointerType+eventType, and then eventType.
 *
 * *** Event Dispatch
 *
 * Events have two methods that will cause early termination: event.abort() will cause no more listeners to be notified
 * for this event, and event.handle() will allow the current level of listeners to be notified (all pointer listeners,
 * or all listeners attached to the current node), but no more listeners after that level will fire. handle and abort
 * are like stopPropagation, stopImmediatePropagation for DOM events, except they do not trigger those DOM methods on
 * the underlying DOM event.
 *
 * Up/down/cancel events all happen separately, but for move events, a specific sequence of events occurs if the pointer
 * changes the node it is over:
 *
 * 1. The move event is fired (and bubbles).
 * 2. An out event is fired for the old topmost Node (and bubbles).
 * 3. exit events are fired for all Nodes in the Trail hierarchy that are now not under the pointer, from the root-most
 *    to the leaf-most. Does not bubble.
 * 4. enter events are fired for all Nodes in the Trail hierarchy that were not under the pointer (but now are), from
 *    the leaf-most to the root-most. Does not bubble.
 * 5. An over event is fired for the new topmost Node (and bubbles).
 *
 * event.abort() and event.handle() will currently not affect other stages in the 'move' sequence (e.g. event.abort() in
 * the 'move' event will not affect the following 'out' event).
 *
 * For each event type:
 *
 * 1. Listeners on the pointer will be triggered first (in the order they were added)
 * 2. Listeners on the target (top-most) Node will be triggered (in the order they were added to that Node)
 * 3. Then if the event bubbles, each Node in the Trail will be triggered, starting from the Node under the top-most
 *    (that just had listeners triggered) and all the way down to the Scene. Listeners are triggered in the order they
 *    were added for each Node.
 *
 * For each listener being notified, it will fire the more specific pointerType+eventType first (e.g. 'mousemove'),
 * then eventType next (e.g. 'move').
 *
 * Currently, preventDefault() is called on the associated DOM event if the top-most node has the 'interactive' property
 * set to a truthy value.
 *
 * *** Relevant Specifications
 *
 * DOM Level 3 events spec: http://www.w3.org/TR/DOM-Level-3-Events/
 * Touch events spec: http://www.w3.org/TR/touch-events/
 * Pointer events spec draft: https://dvcs.w3.org/hg/pointerevents/raw-file/tip/pointerEvents.html
 *                            http://msdn.microsoft.com/en-us/library/ie/hh673557(v=vs.85).aspx
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
  var Event = require( 'SCENERY/input/Event' );
  var BatchedDOMEvent = require( 'SCENERY/input/BatchedDOMEvent' );
  var Emitter = require( 'AXON/Emitter' );

  // Object literal makes it easy to check for the existence of an attribute (compared to [].indexOf()>=0)
  var domEventPropertiesToSerialize = {
    button: true, keyCode: true,
    deltaX: true, deltaY: true, deltaZ: true, deltaMode: true, pointerId: true,
    pointerType: true, charCode: true, which: true, clientX: true, clientY: true, changedTouches: true
  };

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

    this.batchedEvents = [];

    //Pointer for mouse, only created lazily on first mouse event, so no mouse is allocated on tablets
    this.mouse = null;

    this.pointers = [];

    // For PhET-iO
    this.emitter = new Emitter();

    this.pointerAddedListeners = [];

    var self = this;

    // unique to this input instance
    this.onpointerdown = function onpointerdown( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, self.pointerDown, false ); };
    this.onpointerup = function onpointerup( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, self.pointerUp, true ); };
    this.onpointermove = function onpointermove( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, self.pointerMove, false ); };
    this.onpointerover = function onpointerover( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, self.pointerOver, false ); };
    this.onpointerout = function onpointerout( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, self.pointerOut, false ); };
    this.onpointercancel = function onpointercancel( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.POINTER_TYPE, self.pointerCancel, false ); };
    this.onMSPointerDown = function onMSPointerDown( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, self.pointerDown, false ); };
    this.onMSPointerUp = function onMSPointerUp( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, self.pointerUp, true ); };
    this.onMSPointerMove = function onMSPointerMove( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, self.pointerMove, false ); };
    this.onMSPointerOver = function onMSPointerOver( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, self.pointerOver, false ); };
    this.onMSPointerOut = function onMSPointerOut( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, self.pointerOut, false ); };
    this.onMSPointerCancel = function onMSPointerCancel( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MS_POINTER_TYPE, self.pointerCancel, false ); };
    this.ontouchstart = function ontouchstart( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, self.touchStart, false ); };
    this.ontouchend = function ontouchend( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, self.touchEnd, true ); };
    this.ontouchmove = function ontouchmove( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, self.touchMove, false ); };
    this.ontouchcancel = function ontouchcancel( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.TOUCH_TYPE, self.touchCancel, false ); };
    this.onmousedown = function onmousedown( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, self.mouseDown, false ); };
    this.onmouseup = function onmouseup( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, self.mouseUp, true ); };
    this.onmousemove = function onmousemove( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, self.mouseMove, false ); };
    this.onmouseover = function onmouseover( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, self.mouseOver, false ); };
    this.onmouseout = function onmouseout( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.MOUSE_TYPE, self.mouseOut, false ); };
    this.onwheel = function onwheel( domEvent ) { self.batchEvent( domEvent, BatchedDOMEvent.WHEEL_TYPE, self.wheel, false ); };
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
      var self = this;

      var i = this.pointers.length;
      while ( i-- ) {
        var pointer = this.pointers[ i ];
        if ( pointer.point ) {
          var changed = self.branchChangeEvents( pointer, null, false );
          if ( changed ) {
            sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'branch change due validatePointers' );
          }
        }
      }
    },

    dispatchEvent: function( trail, type, pointer, event, bubbles ) {
      sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
        'Input: ' + type + ' on ' + trail.toString() + ' for pointer ' + pointer.toString() + ' at ' + pointer.point.toString() );
      assert && assert( trail, 'Falsy trail for dispatchEvent' );

      // NOTE: event is not immutable, as its currentTarget changes
      var inputEvent = new Event( trail, type, pointer, event );

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

      for ( var i = trail.getLastInputEnabledIndex(); i >= 0; bubbles ? i-- : i = -1 ) {
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
  }, {
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
    KEY_ESCAPE: 27,
    KEY_DELETE: 46,
    KEY_BACKSPACE: 8
  } );
} );
