// Copyright 2002-2012, University of Colorado

/**
 * API for handling mouse / touch / keyboard events.
 *
 * A 'finger' is an abstract way of describing either the mouse, a single touch point, or a key being pressed.
 * touch points and key presses go away after being released, whereas the mouse 'finger' is persistent.
 *
 * DOM Level 3 events spec: http://www.w3.org/TR/DOM-Level-3-Events/
 * Touch events spec: http://www.w3.org/TR/touch-events/
 *
 * @author Jonathan Olson
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
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
      var target = this.scene.root.nodeUnderPoint( finger.point );
      var newPath = target !== null ? target.getPathToRoot() : [];
      var oldPath = finger.path || [];
      
      this.dispatchEvent( newPath, 'up', finger, event );
      
      finger.path = newPath;
    },
    
    downEvent: function( finger, event ) {
      var target = this.scene.root.nodeUnderPoint( finger.point );
      var newPath = target !== null ? target.getPathToRoot() : [];
      var oldPath = finger.path || [];
      
      this.dispatchEvent( newPath, 'down', finger, event );
      
      finger.path = newPath;
    },
    
    moveEvent: function( finger, event ) {
      var target = this.scene.root.nodeUnderPoint( finger.point );
      var newPath = target !== null ? target.getPathToRoot() : [];
      var oldPath = finger.path || [];
      
      var branchIndex;
      
      for ( branchIndex = 0; branchIndex < Math.min( newPath.length, oldPath.length ); branchIndex++ ) {
        if ( newPath[branchIndex] !== oldPath[branchIndex] ) {
          break;
        }
      }
      
      // TODO: if a node gets moved down 1 depth, it may see both an exit and enter?
      if ( oldPath.length > branchIndex ) {
        this.dispatchEvent( oldPath.slice( branchIndex ), 'exit', finger, event );
      }
      if ( newPath.length > branchIndex ) {
        this.dispatchEvent( newPath.slice( branchIndex ), 'enter', finger, event );
      }
      
      this.dispatchEvent( newPath, 'move', finger, event );
      
      finger.path = newPath;
    },
    
    cancelEvent: function( finger, event ) {
      var target = this.scene.root.nodeUnderPoint( finger.point );
      var newPath = target !== null ? target.getPathToRoot() : [];
      var oldPath = finger.path || [];
      
      this.dispatchEvent( newPath, 'cancel', finger, event );
      
      finger.path = newPath;
    },
    
    // targets should be a subpath from a node to an ancestor
    dispatchEvent: function( targets, type, finger, event ) {
      // first run through the finger's listeners to see if one of them will handle the event
      this.dispatchToFinger( type, finger, event );
      
      // if not yet handled, run through the list of targets in order to see if one of them will handle the event
      this.dispatchToTargets( targets, type, finger, event );
      
      // if not yet handled, run through the scene's listeners
      this.dispatchToScene( type, finger, event );
    },
    
    dispatchToFinger: function( type, finger, event ) {
      var fingerListeners = finger.listeners.slice( 0 ); // defensive copy
      for ( var i = 0; i < fingerListeners.length; i++ ) {
        var listener = fingerListeners[i];
        
        if ( listener[type] ) {
          // if a listener returns true, don't handle any more
          var handled = !!( listener[type]( finger, event ) );
          
          if ( handled ) {
            return;
          }
        }
      }
    },
    
    dispatchToTargets: function( targets, type, finger, event ) {
      for ( var i = targets.length - 1; i >= 0; i-- ) {
        var target = targets[i];
        
        var listeners = target.getInputListeners();
        
        for ( var k = 0; k < listeners.length; k++ ) {
          var listener = listeners[k];
          
          if ( listener[type] ) {
            // if a listener returns true, don't handle any more
            var handled = !!( listener[type]( finger, event ) );
            
            if ( handled ) {
              return;
            }
          }
        }
      }
    },
    
    dispatchToScene: function( type, finger, event ) {
      var sceneListeners = this.scene.getInputListeners();
      for ( var i = 0; i < sceneListeners.length; i++ ) {
        var listener = sceneListeners[i];
        
        if ( listener[type] ) {
          // if a listener returns true, don't handle any more
          var handled = !!( listener[type]( finger, event ) );
          
          if ( handled ) {
            return;
          }
        }
      }
    },
  };
  
  Input.Finger = function() {
    this.listeners = [];
  };
  var Finger = Input.Finger;
  Finger.prototype = {
    constructor: Finger,
    
    addListener: function( listener ) {
      phet.assert( !_.contains( this.listeners, listener ) );
      
      this.listeners.push( listener );
    },
    
    removeListener: function( listener ) {
      var index = _.indexOf( this.listeners, listener );
      phet.assert( index !== -1 );
      
      this.listeners.splice( index, 1 );
    }
  };
  
  // track the mouse state
  Input.Mouse = function() {
    Finger.call( this );
    
    this.point = null;
    
    this.leftDown = false;
    this.middleDown = false;
    this.rightDown = false;
    
    this.isMouse = true;
    
    this.path = null;
  };
  var Mouse = Input.Mouse;
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
  
  Input.Touch = function( id, point, event ) {
    Finger.call( this );
    
    this.id = id;
    this.point = point;
    this.isTouch = true;
    this.path = null;
  };
  var Touch = Input.Touch;
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
  
  Input.Key = function( key, event ) {
    Finger.call( this );
    
    this.key = key;
    this.isKey = true;
    this.path = null;
  };
  var Key = Input.Key;
  Key.prototype = _.extend( {}, Finger.prototype, {
    constructor: Key
  } );
  
})();
