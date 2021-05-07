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
 * Event listeners are added with node.addInputListener( listener ), pointer.addInputListener( listener ) and
 * display.addInputListener( listener ).
 * This listener can be an arbitrary object, and the listener will be triggered by calling listener[eventType]( event ),
 * where eventType is one of the event types as described below, and event is a Scenery event with the
 * following properties:
 * - trail {Trail} - Points to the node under the pointer
 * - pointer {Pointer} - The pointer that triggered the event. Additional information about the mouse/touch/pen can be
 *                       obtained from the pointer, for example event.pointer.point.
 * - type {string} - The base type of the event (e.g. for touch down events, it will always just be "down").
 * - domEvent {UIEvent} - The underlying DOM event that triggered this Scenery event. The DOM event may correspond to
 *                        multiple Scenery events, particularly for touch events. This could be a TouchEvent,
 *                        PointerEvent, MouseEvent, MSPointerEvent, etc.
 * - target {Node} - The leaf-most Node in the trail.
 * - currentTarget {Node} - The Node to which the listener being fired is attached, or null if the listener is being
 *                          fired directly from a pointer.
 *
 * Additionally, listeners may support an interrupt() method that detaches it from pointers, or may support being
 * "attached" to a pointer (indicating a primary role in controlling the pointer's behavior). See Pointer for more
 * information about these interactions.
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
 * 4. Listeners on the display will be triggered (in the order they were added)
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

  var BatchedDOMEvent = require( 'SCENERY/input/BatchedDOMEvent' );
  var BrowserEvents = require( 'SCENERY/input/BrowserEvents' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Emitter = require( 'AXON/Emitter' );
  var Event = require( 'SCENERY/input/Event' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Mouse = require( 'SCENERY/input/Mouse' );
  var Pen = require( 'SCENERY/input/Pen' );
  var platform = require( 'PHET_CORE/platform' );
  var scenery = require( 'SCENERY/scenery' );
  var Touch = require( 'SCENERY/input/Touch' );
  var Trail = require( 'SCENERY/util/Trail' );
  var Vector2 = require( 'DOT/Vector2' );

  // constants
  var NORMAL_FREQUENCY = { highFrequency: false };
  var HIGH_FREQUENCY = { highFrequency: true };

  // Object literal makes it easy to check for the existence of an attribute (compared to [].indexOf()>=0)
  var domEventPropertiesToSerialize = {
    button: true, keyCode: true,
    deltaX: true, deltaY: true, deltaZ: true, deltaMode: true, pointerId: true,
    pointerType: true, charCode: true, which: true, clientX: true, clientY: true, changedTouches: true
  };

  function Input( display, attachToWindow, batchDOMEvents, assumeFullWindow, passiveEvents ) {
    assert && assert( display instanceof scenery.Display );
    assert && assert( typeof attachToWindow === 'boolean' );
    assert && assert( typeof batchDOMEvents === 'boolean' );
    assert && assert( typeof assumeFullWindow === 'boolean' );

    this.display = display;
    this.rootNode = display.rootNode;
    this.attachToWindow = attachToWindow;
    this.batchDOMEvents = batchDOMEvents;
    this.assumeFullWindow = assumeFullWindow;
    this.passiveEvents = passiveEvents;
    this.displayUpdateOnEvent = false;

    this.batchedEvents = [];

    //Pointer for mouse, only created lazily on first mouse event, so no mouse is allocated on tablets
    this.mouse = null;

    this.pointers = [];

    // For PhET-iO
    // TODO: this could be made a general thing
    this.emitter = new Emitter();

    this.pointerAddedListeners = [];
  }

  scenery.register( 'Input', Input );

  inherit( Object, Input, {
    /**
     * Interrupts any input actions that are currently taking place (should stop drags, etc.)
     * @public
     */
    interruptPointers: function() {
      _.each( this.pointers, function( pointer ) {
        pointer.interruptAll();
      } );
    },

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

      // Always preventDefault on touch events, since we don't want mouse events triggered afterwards. See
      // http://www.html5rocks.com/en/mobile/touchandmouse/ for more information.
      // Additionally, IE had some issues with skipping prevent default, see
      // https://github.com/phetsims/scenery/issues/464 for mouse handling.
      if ( !( this.passiveEvents === true ) && ( callback !== this.mouseDown || platform.ie || platform.edge ) ) {
        domEvent.preventDefault();
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

    /**
     * Removes all non-Mouse pointers from internal tracking.
     * @public (scenery-internal)
     */
    removeTemporaryPointers: function() {
      var fakeDomEvent = {
        // TODO: Does this break anything
      };

      for ( var i = this.pointers.length - 1; i >= 0; i-- ) {
        var pointer = this.pointers[ i ];
        if ( !( pointer instanceof Mouse ) ) {
          this.pointers.splice( i, 1 );

          // Send exit events. As we can't get a DOM event, we'll send a fake object instead.
          //TODO: consider exit() not taking an event?
          var exitTrail = pointer.trail || new Trail( this.rootNode );
          this.exitEvents( pointer, fakeDomEvent, exitTrail, 0, true );
        }
      }
    },

    connectListeners: function() {
      BrowserEvents.addDisplay( this.display, this.attachToWindow, this.passiveEvents );
    },

    disconnectListeners: function() {
      BrowserEvents.removeDisplay( this.display, this.attachToWindow, this.passiveEvents );
    },

    pointFromEvent: function( domEvent ) {
      var position = Vector2.createFromPool( domEvent.clientX, domEvent.clientY );
      if ( !this.assumeFullWindow ) {
        var domBounds = this.display.domElement.getBoundingClientRect();

        // TODO: consider totally ignoring any with zero width/height, as we aren't attached to the display?
        // For now, don't offset.
        if ( domBounds.width > 0 && domBounds.height > 0 ) {
          position.subtractXY( domBounds.left, domBounds.top );

          // Detect a scaling of the display here (the client bounding rect having different dimensions from our
          // display), and attempt to compensate.
          // NOTE: We can't handle rotation here.
          if ( domBounds.width !== this.display.width || domBounds.height !== this.display.height ) {
            // TODO: Have code verify the correctness here, and that it's not triggering all the time
            position.x *= this.display.width / domBounds.width;
            position.y *= this.display.height / domBounds.height;
          }
        }
      }
      return position;
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

    findPointerById: function( id ) {
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
      this.mouse = new Mouse();
      this.addPointer( this.mouse );
    },

    mouseDown: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseDown(' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'mouseDown', {
          point: { x: point.x, y: point.y },
          event: Input.serializeDomEvent( event )
        }, NORMAL_FREQUENCY );
      }
      if ( !this.mouse ) { this.initMouse(); }

      this.mouse.id = id;
      var pointChanged = this.mouse.down( point, event );
      if ( pointChanged ) {
        this.moveEvent( this.mouse, event );
      }
      this.downEvent( this.mouse, event );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    mouseUp: function( point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseUp(' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'mouseUp', {
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, NORMAL_FREQUENCY );
      }
      if ( !this.mouse ) { this.initMouse(); }

      this.mouse.id = null;
      var pointChanged = this.mouse.up( point, event );
      if ( pointChanged ) {
        this.moveEvent( this.mouse, event );
      }
      this.upEvent( this.mouse, event );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    mouseMove: function( point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseMove(' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'mouseMove', {
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, HIGH_FREQUENCY );
      }
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.move( point, event );
      this.moveEvent( this.mouse, event );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    mouseOver: function( point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseOver(' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'mouseOver', {
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, NORMAL_FREQUENCY );
      }
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.over( point, event );
      // TODO: how to handle mouse-over (and log it)

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    mouseOut: function( point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'mouseOut(' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'mouseOut', {
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, NORMAL_FREQUENCY );
      }
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.out( point, event );
      // TODO: how to handle mouse-out (and log it)

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    // called on mouse wheels
    wheel: function( event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'wheel(' + Input.debugKeyEvent( event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'wheel', {
          event: Input.serializeDomEvent( event )
        }, HIGH_FREQUENCY );
      }
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.wheel( event );

      // don't send mouse-wheel events if we don't yet have a mouse location!
      // TODO: Can we set the mouse location based on the wheel event?
      if ( this.mouse.point ) {
        var trail = this.rootNode.trailUnderPointer( this.mouse ) || new Trail( this.rootNode );
        this.dispatchEvent( trail, 'wheel', this.mouse, event, true );
      }

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    // called for each touch point
    touchStart: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchStart(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'touchStart', {
          id: id,
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, NORMAL_FREQUENCY );
      }
      var touch = new Touch( id, point, event );
      this.addPointer( touch );
      this.downEvent( touch, event );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    touchEnd: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchEnd(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'touchEnd', {
          id: id,
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, NORMAL_FREQUENCY );
      }
      var touch = this.findPointerById( id );
      if ( touch ) {
        var pointChanged = touch.end( point, event );
        if ( pointChanged ) {
          this.moveEvent( touch, event );
        }
        this.removePointer( touch );
        this.upEvent( touch, event );
      }

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    touchMove: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchMove(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'touchMove', {
          id: id,
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, HIGH_FREQUENCY );
      }
      var touch = this.findPointerById( id );
      if ( touch ) {
        touch.move( point, event );
        this.moveEvent( touch, event );
      }

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    touchCancel: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'touchCancel(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'touchCancel', {
          id: id,
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, NORMAL_FREQUENCY );
      }
      var touch = this.findPointerById( id );
      if ( touch ) {
        var pointChanged = touch.cancel( point, event );
        if ( pointChanged ) {
          this.moveEvent( touch, event );
        }
        this.removePointer( touch );
        this.cancelEvent( touch, event );
      }

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    // called for each touch point
    penStart: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penStart(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'penStart', {
          id: id,
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, NORMAL_FREQUENCY );
      }
      var pen = new Pen( id, point, event );
      this.addPointer( pen );
      this.downEvent( pen, event );

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    penEnd: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penEnd(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'penEnd', {
          id: id,
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, NORMAL_FREQUENCY );
      }
      var pen = this.findPointerById( id );
      if ( pen ) {
        var pointChanged = pen.end( point, event );
        if ( pointChanged ) {
          this.moveEvent( pen, event );
        }
        this.removePointer( pen );
        this.upEvent( pen, event );
      }

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    penMove: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penMove(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'penMove', {
          id: id,
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, HIGH_FREQUENCY );
      }
      var pen = this.findPointerById( id );
      if ( pen ) {
        pen.move( point, event );
        this.moveEvent( pen, event );
      }

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    penCancel: function( id, point, event ) {
      sceneryLog && sceneryLog.Input && sceneryLog.Input( 'penCancel(\'' + id + '\',' + Input.debugText( point, event ) + ');' );
      sceneryLog && sceneryLog.Input && sceneryLog.push();

      if ( this.emitter.hasListeners() ) {
        this.emitter.emit3( 'penCancel', {
          id: id,
          point: point.toStateObject(),
          event: Input.serializeDomEvent( event )
        }, NORMAL_FREQUENCY );
      }
      var pen = this.findPointerById( id );
      if ( pen ) {
        var pointChanged = pen.cancel( point, event );
        if ( pointChanged ) {
          this.moveEvent( pen, event );
        }
        this.removePointer( pen );
        this.cancelEvent( pen, event );
      }

      sceneryLog && sceneryLog.Input && sceneryLog.pop();
    },

    pointerDown: function( id, type, point, event ) {
      // In IE for pointer down events, we want to make sure than the next interactions off the page are sent to
      // this element (it will bubble). See https://github.com/phetsims/scenery/issues/464 and
      // http://news.qooxdoo.org/mouse-capturing.
      var target = this.attachToWindow ? document.body : this.display.domElement;
      if ( target.setPointerCapture && event.pointerId ) {
        target.setPointerCapture( event.pointerId );
      }

      switch( type ) {
        case 'mouse':
          // The actual event afterwards
          this.mouseDown( id, point, event );
          break;
        case 'touch':
          this.touchStart( id, point, event );
          break;
        case 'pen':
          this.penStart( id, point, event );
          break;
        default:
          if ( assert ) {
            throw new Error( 'Unknown pointer type: ' + type );
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
          if ( assert ) {
            throw new Error( 'Unknown pointer type: ' + type );
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

    /**
     * Handles a gotpointercapture event, forwarding it to the proper logical event.
     * @public (scenery-internal)
     *
     * @param {number} id
     * @param {string} type
     * @param {Vector2} point
     * @param {Event} event
     */
    gotPointerCapture( id, type, point, event ) {
      const pointer = this.findPointerById( id );

      if ( pointer ) {
        pointer.onGotPointerCapture();
      }
    },

    /**
     * Handles a lostpointercapture event, forwarding it to the proper logical event.
     * @public (scenery-internal)
     *
     * @param {number} id
     * @param {string} type
     * @param {Vector2} point
     * @param {Event} event
     */
    lostPointerCapture( id, type, point, event ) {
      const pointer = this.findPointerById( id );

      if ( pointer ) {
        pointer.onLostPointerCapture();
      }
    },

    /**
     * Handles a pointerover event, forwarding it to the proper logical event.
     * @public (scenery-internal)
     *
     * @param {number} id
     * @param {string} type
     * @param {Vector2} point
     * @param {DOMEvent} event
     */
    pointerOver( id, type, point, event ) {
      // TODO: accumulate mouse/touch info in the object if needed?
      // TODO: do we want to branch change on these types of events?
    },

    pointerOut: function( id, type, point, event ) {

    },

    pointerEnter: function( id, type, point, event ) {

    },

    pointerLeave: function( id, type, point, event ) {

    },

    upEvent: function( pointer, event ) {
      var trail = this.rootNode.trailUnderPointer( pointer ) || new Trail( this.rootNode );

      this.dispatchEvent( trail, 'up', pointer, event, true );

      // touch pointers are transient, so fire exit/out to the trail afterwards
      if ( pointer instanceof Touch ) {
        this.exitEvents( pointer, event, trail, 0, true );
      }

      pointer.trail = trail;
    },

    downEvent: function( pointer, event ) {
      var trail = this.rootNode.trailUnderPointer( pointer ) || new Trail( this.rootNode );

      // touch pointers are transient, so fire enter/over to the trail first
      if ( pointer instanceof Touch ) {
        this.enterEvents( pointer, event, trail, 0, true );
      }

      this.dispatchEvent( trail, 'down', pointer, event, true );

      pointer.trail = trail;

      // a11y
      var focusableNode = null;
      var trailAccessible = !trail.rootNode()._rendererSummary.isNotAccessible();

      // If any node in the trail has accessible content
      if ( trailAccessible ) {

        // Starting with the leaf most node, search for the closest accessible ancestor from the node under the pointer.
        for ( var i = trail.nodes.length - 1; i >= 0; i-- ) {
          if ( trail.nodes[ i ].focusable ) {
            focusableNode = trail.nodes[ i ];
            break;
          }
        }

        // Remove keyboard focus, but store element that is receiving interaction in case we resume .
        this.display.pointerFocus = focusableNode;
        scenery.Display.focus = null;
      }
    },

    moveEvent: function( pointer, event ) {
      var changed = this.branchChangeEvents( pointer, event, true );
      if ( changed ) {
        sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent( 'branch change due to move event' );
      }
    },

    cancelEvent: function( pointer, event ) {
      var trail = this.rootNode.trailUnderPointer( pointer ) || new Trail( this.rootNode );

      this.dispatchEvent( trail, 'cancel', pointer, event, true );

      // touch pointers are transient, so fire exit/out to the trail afterwards
      if ( pointer instanceof Touch ) {
        this.exitEvents( pointer, event, trail, 0, true );
      }

      pointer.trail = trail;
    },

    // return whether there was a change
    branchChangeEvents: function( pointer, event, isMove ) {
      var trail = this.rootNode.trailUnderPointer( pointer ) || new Trail( this.rootNode );
      sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
        'checking branch change: ' + trail.toString() + ' at ' + pointer.point.toString() );
      var oldTrail = pointer.trail || new Trail( this.rootNode ); // TODO: consider a static trail reference

      var lastNodeChanged = oldTrail.lastNode() !== trail.lastNode();

      var branchIndex = Trail.branchIndex( trail, oldTrail );
      var isBranchChange = branchIndex !== trail.length || branchIndex !== oldTrail.length;
      isBranchChange && sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
        'branch change from ' + oldTrail.toString() + ' to ' + trail.toString() );

      // event order matches http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
      if ( isMove ) {
        this.dispatchEvent( trail, 'move', pointer, event, true );
      }

      // we want to approximately mimic http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
      // TODO: if a node gets moved down 1 depth, it may see both an exit and enter?
      if ( isBranchChange ) {
        this.exitEvents( pointer, event, oldTrail, branchIndex, lastNodeChanged );
        this.enterEvents( pointer, event, trail, branchIndex, lastNodeChanged );
      }

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

    /**
     * Dispatch to all nodes in the Trail, optionally bubbling down from the leaf to the root.
     * @private
     *
     * @param {Trail} trail
     * @param {string} type
     * @param {Pointer} pointer
     * @param {DOMEvent} event
     * @param {boolean} bubbles - If bubbles is false, the event is only dispatched to the leaf node of the trail.
     */
    dispatchEvent: function( trail, type, pointer, event, bubbles ) {
      sceneryLog && sceneryLog.InputEvent && sceneryLog.InputEvent(
        'Input: ' + type + ' on ' + trail.toString() + ' for pointer ' + pointer.toString() + ' at ' + pointer.point.toString() );
      assert && assert( trail, 'Falsy trail for dispatchEvent' );

      // NOTE: event is not immutable, as its currentTarget changes
      var inputEvent = new Event( trail, type, pointer, event );

      // first run through the pointer's listeners to see if one of them will handle the event
      this.dispatchToListeners( pointer, pointer.getListeners(), type, inputEvent );

      // if not yet handled, run through the trail in order to see if one of them will handle the event
      // at the base of the trail should be the scene node, so the scene will be notified last
      this.dispatchToTargets( trail, type, pointer, inputEvent, bubbles );

      // Notify input listeners on the Display
      this.dispatchToListeners( pointer, this.display.getInputListeners(), type, inputEvent );
    },

    /**
     * Notifies an array of listeners with a specific event.
     * @private
     *
     * @param {Pointer} pointer
     * @param {Array.<Object>} listeners - Should be a defensive array copy already.
     * @param {string} type
     * @param {Event} inputEvent
     */
    dispatchToListeners: function( pointer, listeners, type, inputEvent ) {
      if ( inputEvent.handled ) {
        return;
      }

      var specificType = pointer.type + type; // e.g. mouseup, touchup

      for ( var i = 0; i < listeners.length; i++ ) {
        var listener = listeners[ i ];

        ( !inputEvent.aborted && listener[ specificType ] ) && listener[ specificType ]( inputEvent );
        ( !inputEvent.aborted && listener[ type ] ) && listener[ type ]( inputEvent );
      }
    },

    /**
     * Dispatch to all nodes in the Trail, optionally bubbling down from the leaf to the root.
     * @private
     *
     * @param {Trail} trail
     * @param {string} type
     * @param {Pointer} pointer
     * @param {Event} inputEvent
     * @param {boolean} bubbles - If bubbles is false, the event is only dispatched to the leaf node of the trail.
     */
    dispatchToTargets: function( trail, type, pointer, inputEvent, bubbles ) {
      if ( inputEvent.aborted || inputEvent.handled ) {
        return;
      }

      for ( var i = trail.getLastInputEnabledIndex(); i >= 0; bubbles ? i-- : i = -1 ) {
        var target = trail.nodes[ i ];
        if ( target.isDisposed() ) {
          continue;
        }

        inputEvent.currentTarget = target;

        this.dispatchToListeners( pointer, target.getInputListeners(), type, inputEvent );

        // if the input event was aborted or handled, don't follow the trail down another level
        if ( inputEvent.aborted || inputEvent.handled ) {
          return;
        }
      }
    },

    // @public (phet-io)
    invokeInputEvent: function( command, options ) {
      if ( command === 'mouseMove' ) {this.mouseMove( Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'mouseDown' ) {this.mouseDown( null, Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'mouseUp' ) {this.mouseUp( Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'mouseOver' ) {this.mouseOver( Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'mouseOut' ) {this.mouseOut( Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'wheel' ) {this.wheel( options.event );}
      else if ( command === 'touchStart' ) {this.touchStart( options.id, Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'touchEnd' ) {this.touchEnd( options.id, Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'touchMove' ) {this.touchMove( options.id, Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'touchCancel' ) {this.touchCancel( options.id, Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'penStart' ) {this.penStart( options.id, Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'penEnd' ) {this.penEnd( options.id, Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'penMove' ) {this.penMove( options.id, Vector2.fromStateObject( options.point ), options.event );}
      else if ( command === 'penCancel' ) {this.penCancel( options.id, Vector2.fromStateObject( options.point ), options.event );}
    }
  }, {
    serializeDomEvent: function serializeDomEvent( domEvent ) {
      var entries = {};
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
            entries[ prop ] = arr;
          }
          else {
            entries[ prop ] = ( ( typeof domEvent[ prop ] === 'object' ) && ( domEvent[ prop ] !== null ) ? {} : JSON.parse( JSON.stringify( domEvent[ prop ] ) ) ); // TODO: is parse/stringify necessary?
          }
        }
      }
      return entries;
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
    }
  } );

  Input.BASIC_EVENT_TYPES = [ 'down', 'up', 'cancel', 'move', 'wheel', 'enter', 'exit', 'over', 'out' ];
  Input.EVENT_PREFIXES = [ '', 'mouse', 'touch', 'pen' ];
  Input.ALL_EVENT_TYPES = Input.EVENT_PREFIXES.map( function( prefix ) {
    return Input.BASIC_EVENT_TYPES.map( function( eventName ) {
      return prefix + eventName;
    } );
  } );

  return Input;
} );
