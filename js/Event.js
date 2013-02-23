// Copyright 2002-2012, University of Colorado

define( function( require ) {
  
  var Event = function( options ) {
    this.handled = false;
    this.aborted = false;
    
    // put all properties in the options object into this event
    _.extend( this, options );
    
    // TODO: add extended information based on an event here?
  };
  
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
