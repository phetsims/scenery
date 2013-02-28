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
  
  var scenery = require( 'SCENERY/scenery' );
  
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
  
  return Event;
} );
