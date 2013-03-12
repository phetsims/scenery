// Copyright 2002-2012, University of Colorado

/*
 * An event in Scenery that has similar event-handling characteristics to DOM events.
 * The original DOM event (if any) is available as event.domEvent.
 *
 * Multiple events can be triggered by a single domEvent, so don't assume it is unique.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */
 
define( function( require ) {
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Event = function( arguments ) {
    // ensure that all of the required arguments are supplied
    assert && assert( arguments.trail &&
                      arguments.type &&
                      arguments.pointer &&
                      arguments.domEvent &&
                      arguments.target, 'Missing required scenery.Event argument' );
    
    this.handled = false;
    this.aborted = false;
    
    // {Trail} path to the leaf-most node, ordered list, from root to leaf
    this.trail = arguments.trail;
    
    // {String} what event was triggered on the listener
    this.type = arguments.type;
    
    // {Pointer}
    this.pointer = arguments.pointer;
    
    // raw DOM InputEvent (TouchEvent, PointerEvent, MouseEvent,...)
    this.domEvent = arguments.domEvent;
    
    // {Node} whatever node you attached the listener to, or null when firing events on a Pointer
    this.currentTarget = arguments.currentTarget;
    
    // {Node} leaf-most node in trail
    this.target = arguments.trail;
    
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
  
  return Event;
} );
