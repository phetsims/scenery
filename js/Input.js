// Copyright 2002-2012, University of Colorado

/**
 * API for handling mouse / touch / keyboard events.
 *
 * A 'finger' is an abstract way of describing either the mouse, a single touch point, or a key being pressed.
 * touch points and key presses go away after being released, whereas the mouse 'finger' is persistent.
 *
 * DOM Level 3 events spec: http://www.w3.org/TR/DOM-Level-3-Events/
 * Touch events spec: http://www.w3.org/TR/touch-events/
 * Pointer events spec draft: https://dvcs.w3.org/hg/pointerevents/raw-file/tip/pointerEvents.html
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  scenery.Input = function( scene ) {
    this.scene = scene;
    
    this.mouse = new Mouse();
    
    this.fingers = [ this.mouse ];
  };
  
  var Input = scenery.Input;
  
  Input.prototype = {
    constructor: Input,
    
    addFinger: function( finger ) {
      this.fingers.push( finger );
    },
    
    removeFinger: function( finger ) {
      // sanity check version, will remove all instances
      for ( var i = this.fingers.length - 1; i >= 0; i-- ) {
        if ( this.fingers[i] === finger ) {
          this.fingers.splice( i, 1 );
        }
      }
    },
    
    findTouchById: function( id ) {
      return _.find( this.fingers, function( finger ) { return finger.id === id; } );
    },
    
    mouseDown: function( point, event ) {
      this.mouse.down( point, event );
      this.downEvent( this.mouse, event );
    },
    
    mouseUp: function( point, event ) {
      this.mouse.up( point, event );
      this.upEvent( this.mouse, event );
    },
    
    mouseMove: function( point, event ) {
      this.mouse.move( point, event );
      this.moveEvent( this.mouse, event );
    },
    
    mouseOver: function( point, event ) {
      this.mouse.over( point, event );
      // TODO: how to handle mouse-over
    },
    
    mouseOut: function( point, event ) {
      this.mouse.out( point, event );
      // TODO: how to handle mouse-out
    },
    
    // called for each touch point
    touchStart: function( id, point, event ) {
      var touch = new Touch( id, point, event );
      this.addFinger( touch );
      this.downEvent( touch, event );
    },
    
    touchEnd: function( id, point, event ) {
      var touch = this.findTouchById( id );
      touch.end( point, event );
      this.removeFinger( touch );
      this.upEvent( touch, event );
    },
    
    touchMove: function( id, point, event ) {
      var touch = this.findTouchById( id );
      touch.move( point, event );
      this.moveEvent( touch, event );
    },
    
    touchCancel: function( id, point, event ) {
      var touch = this.findTouchById( id );
      touch.cancel( point, event );
      this.removeFinger( touch );
      this.cancelEvent( touch, event );
    },
    
    
    upEvent: function( finger, event ) {
      var trail = this.scene.trailUnderPoint( finger.point ) || new scenery.Trail( this.scene );
      
      this.dispatchEvent( trail, 'up', finger, event, true );
      
      finger.trail = trail;
    },
    
    downEvent: function( finger, event ) {
      var trail = this.scene.trailUnderPoint( finger.point ) || new scenery.Trail( this.scene );
      
      this.dispatchEvent( trail, 'down', finger, event, true );
      
      finger.trail = trail;
    },
    
    moveEvent: function( finger, event ) {
      var trail = this.scene.trailUnderPoint( finger.point ) || new scenery.Trail( this.scene );
      var oldTrail = finger.trail || new scenery.Trail( this.scene );
      
      var lastNodeChanged = oldTrail.lastNode() !== trail.lastNode();
      
      var branchIndex;
      
      for ( branchIndex = 0; branchIndex < Math.min( trail.length, oldTrail.length ); branchIndex++ ) {
        if ( trail.nodes[branchIndex] !== oldTrail.nodes[branchIndex] ) {
          break;
        }
      }
      
      if ( lastNodeChanged ) {
        this.dispatchEvent( oldTrail, 'out', finger, event, true );
      }
      
      // we want to approximately mimic http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order
      // TODO: if a node gets moved down 1 depth, it may see both an exit and enter?
      if ( oldTrail.length > branchIndex ) {
        for ( var oldIndex = branchIndex; oldIndex < oldTrail.length; oldIndex++ ) {
          this.dispatchEvent( oldTrail.slice( 0, oldIndex + 1 ), 'exit', finger, event, false );
        }
      }
      if ( trail.length > branchIndex ) {
        for ( var newIndex = branchIndex; newIndex < trail.length; newIndex++ ) {
          this.dispatchEvent( trail.slice( 0, newIndex + 1 ), 'enter', finger, event, false );
        }
      }
      
      if ( lastNodeChanged ) {
        this.dispatchEvent( trail, 'over', finger, event, true );
      }
      
      // TODO: move the 'move' event to before the others, matching http://www.w3.org/TR/DOM-Level-3-Events/#events-mouseevent-event-order ?
      this.dispatchEvent( trail, 'move', finger, event, true );
      
      finger.trail = trail;
    },
    
    cancelEvent: function( finger, event ) {
      var trail = this.scene.trailUnderPoint( finger.point );
      
      this.dispatchEvent( trail, 'cancel', finger, event, true );
      
      finger.trail = trail;
    },
    
    dispatchEvent: function( trail, type, finger, event, bubbles ) {
      // TODO: is there a way to make this event immutable?
      var inputEvent = new Event( {
        trail: trail,
        type: type,
        finger: finger,
        domEvent: event,
        currentTarget: null,
        target: trail.lastNode()
      } );
      
      // first run through the finger's listeners to see if one of them will handle the event
      this.dispatchToFinger( type, finger, inputEvent );
      
      // if not yet handled, run through the trail in order to see if one of them will handle the event
      // at the base of the trail should be the scene node, so the scene will be notified last
      this.dispatchToTargets( trail, type, inputEvent, bubbles );
    },
    
    dispatchToFinger: function( type, finger, inputEvent ) {
      if ( inputEvent.aborted || inputEvent.handled ) {
        return;
      }
      
      var fingerListeners = finger.listeners.slice( 0 ); // defensive copy
      for ( var i = 0; i < fingerListeners.length; i++ ) {
        var listener = fingerListeners[i];
        
        if ( listener[type] ) {
          // if a listener returns true, don't handle any more
          var aborted = !!( listener[type]( inputEvent ) ) || inputEvent.aborted;
          
          if ( aborted ) {
            return;
          }
        }
      }
    },
    
    dispatchToTargets: function( trail, type, inputEvent, bubbles ) {
      if ( inputEvent.aborted || inputEvent.handled ) {
        return;
      }
      
      for ( var i = trail.length - 1; i >= 0; bubbles ? i-- : i = -1 ) {
        var target = trail.nodes[i];
        inputEvent.currentTarget = target;
        
        var listeners = target.getInputListeners();
        
        for ( var k = 0; k < listeners.length; k++ ) {
          var listener = listeners[k];
          
          if ( listener[type] ) {
            // if a listener returns true, don't handle any more
            var aborted = !!( listener[type]( inputEvent ) ) || inputEvent.aborted;
            
            if ( aborted ) {
              return;
            }
          }
        }
        
        // if the input event was handled, don't follow the trail down another level
        if ( inputEvent.handled ) {
          return;
        }
      }
    }
  };
  
  scenery.Event = function( options ) {
    this.handled = false;
    this.aborted = false;
    
    // put all properties in the options object into this event
    _.extend( this, options );
    
    // TODO: add extended information based on an event here?
  };
  var Event = scenery.Event;
  
  Event.prototype = {
    constructor: Event,
    
    // like DOM Event.stopPropagation(), but named differently to indicate it doesn't fire that behavior on the underlying DOM event
    handle: function() {
      this.handled = true;
    },
    
    // like DOM Event.stopImmediatePropagation(), but named differently to indicate it doesn't fire that behavior on the underlying DOM event
    abort: function() {
      this.handled = true;
      this.aborted = true;
    }
  };
  
  /*
   * Conventionally set flags on a finger: TODO: add this state tracking to finger for convenience
   * dragging - whether the finger is dragging something
   *
   * TODO: consider an 'active' flag?
   */
  scenery.Finger = function() {
    this.listeners = [];
  };
  var Finger = scenery.Finger;
  Finger.prototype = {
    constructor: Finger,
    
    addInputListener: function( listener ) {
      assert && assert( !_.contains( this.listeners, listener ) );
      
      this.listeners.push( listener );
    },
    
    removeInputListener: function( listener ) {
      var index = _.indexOf( this.listeners, listener );
      assert && assert( index !== -1 );
      
      this.listeners.splice( index, 1 );
    }
  };
  
  // track the mouse state
  scenery.Mouse = function() {
    Finger.call( this );
    
    this.point = null;
    
    this.leftDown = false;
    this.middleDown = false;
    this.rightDown = false;
    
    this.isMouse = true;
    
    this.trail = null;
  };
  var Mouse = scenery.Mouse;
  Mouse.prototype = _.extend( {}, Finger.prototype, {
    constructor: Mouse,
    
    down: function( point, event ) {
      this.point = point;
      switch( event.button ) {
        case 0: this.leftDown = true; break;
        case 1: this.middleDown = true; break;
        case 2: this.rightDown = true; break;
      }
    },
    
    up: function( point, event ) {
      this.point = point;
      switch( event.button ) {
        case 0: this.leftDown = false; break;
        case 1: this.middleDown = false; break;
        case 2: this.rightDown = false; break;
      }
    },
    
    move: function( point, event ) {
      this.point = point;
    },
    
    over: function( point, event ) {
      this.point = point;
    },
    
    out: function( point, event ) {
      // TODO: how to handle the mouse out-of-bounds
      this.point = null;
    }
  } );
  
  scenery.Touch = function( id, point, event ) {
    Finger.call( this );
    
    this.id = id;
    this.point = point;
    this.isTouch = true;
    this.trail = null;
  };
  var Touch = scenery.Touch;
  Touch.prototype = _.extend( {}, Finger.prototype, {
    constructor: Touch,
    
    move: function( point, event ) {
      this.point = point;
    },
    
    end: function( point, event ) {
      this.point = point;
    },
    
    cancel: function( point, event ) {
      this.point = point;
    }
  } );
  
  scenery.Key = function( key, event ) {
    Finger.call( this );
    
    this.key = key;
    this.isKey = true;
    this.trail = null;
  };
  var Key = scenery.Key;
  Key.prototype = _.extend( {}, Finger.prototype, {
    constructor: Key
  } );
  
} );
