// Copyright 2002-2013, University of Colorado

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
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/input/Mouse' );
  require( 'SCENERY/input/Touch' );
  require( 'SCENERY/input/Pen' );
  require( 'SCENERY/input/Key' );
  require( 'SCENERY/input/Event' );
  
  // listenerTarget is the DOM node (window/document/element) to which DOM event listeners will be attached
  scenery.Input = function Input( scene, listenerTarget, batchDOMEvents ) {
    this.scene = scene;
    this.listenerTarget = listenerTarget;
    this.batchDOMEvents = batchDOMEvents;
    
    this.batchedCallbacks = []; // cleared every frame

    //Pointer for mouse, only created lazily on first mouse event, so no mouse is allocated on tablets
    this.mouse = null;

    this.pointers = [];
    
    this.listenerReferences = [];
    
    this.eventLog = [];     // written when recording event input. can be overwritten to the empty array to reset. Strings relative to this class (prefix "scene.input.")
    this.logEvents = false; // can be set to true to cause Scenery to record all input calls to eventLog

    this.pointerAddedListeners = [];
  };
  var Input = scenery.Input;
  
  Input.prototype = {
    constructor: Input,

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
      this.pointerAddedListeners.push(listener);
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
      assert && assert( event.hasOwnProperty( 'keyCode' ) && event.hasOwnProperty('charCode'), 'Assumes the KeyboardEvent has keyCode and charCode properties' );
      var result = _.find( this.pointers, function( pointer ) {
        // TODO: also check location (if that exists), so we don't mix up left and right shift, etc.
        return pointer.keyCode === event.keyCode && pointer.charCode === event.charCode;
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
      if ( this.logEvents ) { this.eventLog.push( 'mouseDown(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      var pointChanged = this.mouse.down( point, event );
      if ( pointChanged ) {
        this.moveEvent( this.mouse, event );
      }
      this.downEvent( this.mouse, event );
    },
    
    mouseUp: function( point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'mouseUp(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      var pointChanged = this.mouse.up( point, event );
      if ( pointChanged ) {
        this.moveEvent( this.mouse, event );
      }
      this.upEvent( this.mouse, event );
    },
    
    mouseUpImmediate: function( point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'mouseUpImmediate(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      if ( this.mouse.point ) {
        // if the pointer's point hasn't been initialized yet, ignore the immediate up
        this.upImmediateEvent( this.mouse, event );
      }
    },
    
    mouseMove: function( point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'mouseMove(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.move( point, event );
      this.moveEvent( this.mouse, event );
    },
    
    mouseOver: function( point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'mouseOver(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.over( point, event );
      // TODO: how to handle mouse-over (and log it)
    },
    
    mouseOut: function( point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'mouseOut(' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      if ( !this.mouse ) { this.initMouse(); }
      this.mouse.out( point, event );
      // TODO: how to handle mouse-out (and log it)
    },
    
    keyDown: function( event ) {
      if ( this.logEvents ) { this.eventLog.push( 'keyDown(' + Input.serializeDomEvent( event ) + ');' ); }
      var key = new scenery.Key( event );
      this.addPointer( key );
      
      var trail = this.scene.getTrailFromKeyboardFocus();
      this.dispatchEvent( trail, 'keyDown', key, event, true );
    },
    
    keyUp: function( event ) {
      if ( this.logEvents ) { this.eventLog.push( 'keyUp(' + Input.serializeDomEvent( event ) + ');' ); }
      var key = this.findKeyByEvent( event );
      if ( key ) {
        this.removePointer( key );
        
        var trail = this.scene.getTrailFromKeyboardFocus();
        this.dispatchEvent( trail, 'keyUp', key, event, true );
      }
    },
    
    keyPress: function( event ) {
      if ( this.logEvents ) { this.eventLog.push( 'keyPress(' + Input.serializeDomEvent( event ) + ');' ); }
      // NOTE: do we even need keyPress?
    },
    
    // called for each touch point
    touchStart: function( id, point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'touchStart(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var touch = new scenery.Touch( id, point, event );
      this.addPointer( touch );
      this.downEvent( touch, event );
    },
    
    touchEnd: function( id, point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'touchEnd(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var touch = this.findTouchById( id );
      if ( touch ) {
        var pointChanged = touch.end( point, event );
        if ( pointChanged ) {
          this.moveEvent( touch, event );
        }
        this.removePointer( touch );
        this.upEvent( touch, event );
      } else {
        assert && assert( false, 'Touch not found for touchEnd: ' + id );
      }
    },
    
    touchEndImmediate: function( id, point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'touchEndImmediate(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var touch = this.findTouchById( id );
      if ( touch ) {
        this.upImmediateEvent( touch, event );
      } else {
        assert && assert( false, 'Touch not found for touchEndImmediate: ' + id );
      }
    },
    
    touchMove: function( id, point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'touchMove(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var touch = this.findTouchById( id );
      if ( touch ) {
        touch.move( point, event );
        this.moveEvent( touch, event );
      } else {
        assert && assert( false, 'Touch not found for touchMove: ' + id );
      }
    },
    
    touchCancel: function( id, point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'touchCancel(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var touch = this.findTouchById( id );
      if ( touch ) {
        var pointChanged = touch.cancel( point, event );
        if ( pointChanged ) {
          this.moveEvent( touch, event );
        }
        this.removePointer( touch );
        this.cancelEvent( touch, event );
      } else {
        assert && assert( false, 'Touch not found for touchCancel: ' + id );
      }
    },
    
    // called for each touch point
    penStart: function( id, point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'penStart(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var pen = new scenery.Pen( id, point, event );
      this.addPointer( pen );
      this.downEvent( pen, event );
    },
    
    penEnd: function( id, point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'penEnd(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var pen = this.findTouchById( id );
      if ( pen ) {
        var pointChanged = pen.end( point, event );
        if ( pointChanged ) {
          this.moveEvent( pen, event );
        }
        this.removePointer( pen );
        this.upEvent( pen, event );
      } else {
        assert && assert( false, 'Pen not found for penEnd: ' + id );
      }
    },
    
    penEndImmediate: function( id, point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'penEndImmediate(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var pen = this.findTouchById( id );
      if ( pen ) {
        this.upImmediateEvent( pen, event );
      } else {
        assert && assert( false, 'Pen not found for penEndImmediate: ' + id );
      }
    },
    
    penMove: function( id, point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'penMove(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var pen = this.findTouchById( id );
      if ( pen ) {
        pen.move( point, event );
        this.moveEvent( pen, event );
      } else {
        assert && assert( false, 'Pen not found for penMove: ' + id );
      }
    },
    
    penCancel: function( id, point, event ) {
      if ( this.logEvents ) { this.eventLog.push( 'penCancel(\'' + id + '\',' + Input.serializeVector2( point ) + ',' + Input.serializeDomEvent( event ) + ');' ); }
      var pen = this.findTouchById( id );
      if ( pen ) {
        var pointChanged = pen.cancel( point, event );
        if ( pointChanged ) {
          this.moveEvent( pen, event );
        }
        this.removePointer( pen );
        this.cancelEvent( pen, event );
      } else {
        assert && assert( false, 'Pen not found for penCancel: ' + id );
      }
    },
    
    pointerDown: function( id, type, point, event ) {
      switch ( type ) {
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
      switch ( type ) {
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
    
    pointerUpImmediate: function( id, type, point, event ) {
      switch ( type ) {
        case 'mouse':
          this.mouseUpImmediate( point, event );
          break;
        case 'touch':
          this.touchEndImmediate( id, point, event );
          break;
        case 'pen':
          this.penEndImmediate( id, point, event );
          break;
        default:
          if ( console.log ) {
            console.log( 'Unknown pointer type: ' + type );
          }
      }
    },
    
    pointerCancel: function( id, type, point, event ) {
      switch ( type ) {
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
      switch ( type ) {
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
      var trail = this.scene.trailUnderPointer( pointer ) || new scenery.Trail( this.scene );
      
      this.dispatchEvent( trail, 'up', pointer, event, true );
      
      // touch pointers are transient, so fire exit/out to the trail afterwards
      if ( pointer.isTouch ) {
        this.exitEvents( pointer, event, trail, 0, true );
      }
      
      pointer.trail = trail;
    },
    
    upImmediateEvent: function( pointer, event ) {
      var trail = this.scene.trailUnderPointer( pointer ) || new scenery.Trail( this.scene );
      
      this.dispatchEvent( trail, 'upImmediate', pointer, event, true );
    },
    
    downEvent: function( pointer, event ) {
      var trail = this.scene.trailUnderPointer( pointer ) || new scenery.Trail( this.scene );
      
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
      var trail = this.scene.trailUnderPointer( pointer ) || new scenery.Trail( this.scene );
      
      this.dispatchEvent( trail, 'cancel', pointer, event, true );
      
      // touch pointers are transient, so fire exit/out to the trail afterwards
      if ( pointer.isTouch ) {
        this.exitEvents( pointer, event, trail, 0, true );
      }
      
      pointer.trail = trail;
    },
    
    // return whether there was a change
    branchChangeEvents: function( pointer, event, isMove ) {
      var trail = this.scene.trailUnderPointer( pointer ) || new scenery.Trail( this.scene );
      sceneryEventLog && sceneryEventLog( 'checking branch change: ' + trail.toString() + ' at ' + pointer.point.toString() );
      var oldTrail = pointer.trail || new scenery.Trail( this.scene ); // TODO: consider a static trail reference
      
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
        } catch ( e ) {
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
      
      var specificType = pointer.type + type; // e.g. mouseup, touchup, keyup
      
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
      
      var specificType = pointer.type + type; // e.g. mouseup, touchup, keyup
      
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
    },
    
    addListener: function( type, callback, useCapture ) {
      var input = this;
      
      //Cancel propagation of mouse events but not key events.  Key Events need to propagate for tab navigability
      var usePreventDefault = type !== 'keydown' && type !== 'keyup' && type !== 'keypress';
      
      if ( this.batchDOMEvents ) {
        var batchedCallback = function batchedEvent( domEvent ) {
          sceneryEventLog && sceneryEventLog( 'Batching event for ' + type );
          
          if ( usePreventDefault ) {
            domEvent.preventDefault(); // TODO: should we batch the events in a different place so we don't preventDefault on something bad?
          }
          input.batchedCallbacks.push( function batchedEventCallback() {
            // process whether anything under the pointers changed before running additional input events
            sceneryEventLog && sceneryEventLog( 'validatePointers from batched event' );
            input.validatePointers();
            if ( input.logEvents ) { input.eventLog.push( 'validatePointers();' ); }
            
            callback( domEvent );
          } );
        };
        this.listenerTarget.addEventListener( type, batchedCallback, useCapture );
        this.listenerReferences.push( { type: type, callback: batchedCallback, useCapture: useCapture } );
      } else {
        this.listenerTarget.addEventListener( type, callback, useCapture );
        this.listenerReferences.push( { type: type, callback: function synchronousEvent( domEvent ) {
          sceneryEventLog && sceneryEventLog( 'Running event for ' + type );
          
          // process whether anything under the pointers changed before running additional input events
          sceneryEventLog && sceneryEventLog( 'validatePointers from non-batched event' );
          input.validatePointers();
          if ( input.logEvents ) { input.eventLog.push( 'validatePointers();' ); }
          
          callback( domEvent );
        }, useCapture: useCapture } );
      }
    },
    
    // temporary, for mouse events
    addImmediateListener: function( type, callback, useCapture ) {
      var input = this;
      
      this.listenerTarget.addEventListener( type, callback, useCapture );
      this.listenerReferences.push( { type: type, callback: function immediateEvent( domEvent ) {
        sceneryEventLog && sceneryEventLog( 'Running immediate event for ' + type );
        
        // process whether anything under the pointers changed before running additional input events
        // input.validatePointers();
        // if ( input.logEvents ) { input.eventLog.push( 'validatePointers();' ); }
        
        callback( domEvent );
      }, useCapture: useCapture } );
    },
    
    disposeListeners: function() {
      var input = this;
      _.each( this.listenerReferences, function( ref ) {
        input.listenerTarget.removeEventListener( ref.type, ref.callback, ref.useCapture );
      } );
    },
    
    fireBatchedEvents: function() {
      if ( this.batchedCallbacks.length ) {
        sceneryEventLog && sceneryEventLog( 'Input.fireBatchedEvents length:' + this.batchedCallbacks.length );
        var len = this.batchedCallbacks.length;
        for ( var i = 0; i < len; i++ ) {
          this.batchedCallbacks[i]();
        }
        this.batchedCallbacks.length = 0;
      }
    }
  };
  
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
        } else {
          lines.push( prop + ':' + ( ( typeof domEvent[prop] === 'object' ) && ( domEvent[prop] !== null ) ? '{}' : JSON.stringify( domEvent[prop] ) ) );
        }
      }
    }
    return '{' + lines.join( ',' ) + '}';
  };
  
  Input.serializeVector2 = function( vector ) {
    return 'dot(' + vector.x + ',' + vector.y + ')';
  };
  
  return Input;
} );
