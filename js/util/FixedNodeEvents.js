// Copyright 2002-2013, University of Colorado

/**
 * Mix-in for Node's event handling, with experimental performance enhancements
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  var eventNames = [
    'selfBounds',
    'childBounds',
    'localBounds',
    'bounds',
    'resize',
    'boundsAccuracy',
    'transform'
  ];
  
  var eventsRequiringBoundsValidation = {
    'childBounds': true,
    'localBounds': true,
    'bounds': true
  };
  
  scenery.FixedNodeEvents = function FixedNodeEvents( type ) {
    var proto = type.prototype;
    
    // this should be called in the constructor to initialize
    proto.initializeNodeEvents = function() {
      this._events = {};
      
      var node = this;
      
      // TODO: performance: consider delaying this? Could affect memory usage?
      var len = eventNames.length;
      for ( var i = 0; i < len; i++ ) {
        node._events[eventNames[i]] = [];
      }
    };
    
    /**
     * @param {String}   type     The type of event, like 'resize' or 'bounds'
     * @param {Function} listener Callback, called with arguments that depend on the event type
     */
    proto.addEventListener = function( type, listener ) {
      assert && assert( type !== undefined && listener !== undefined,
                                      'Both a type and listener are required for addEventListener' );
      
      // most commonly a bug, maybe there will be a good use case? can always work around by wrapping with a new function each time
      assert && assert( _.indexOf( this._events[type], listener ),
                                      'Event listener was already there for addEventListener with type ' + type );
      
      this._events[type].push( listener );
      
      if ( type in eventsRequiringBoundsValidation ) {
        this.changeBoundsEventCount( 1 );
        this._boundsEventSelfCount++;
      }
      
      // allow chaining
      return this;
    };

    /**
     * Check to see whether this Node contains the specified listener
     * @param {string} type type of listener
     * @param {function} listener the listener instance
     * @returns {boolean} true if the listener is already registered with this Node
     */
    proto.containsEventListener = function( type, listener ) {
      return _.indexOf( this._events[type], listener ) >= 0;
    };

    /**
     * @param {String}   type     The type of event, like 'resize' or 'bounds'
     * @param {Function} listener The callback to remove.
     */
    proto.removeEventListener = function( type, listener ) {
      assert && assert( type !== undefined && listener !== undefined,
                                      'Both a type and listener are required for removeEventListener' );
      
      // ensure the listener is in our list
      assert && assert( _.indexOf( this._events[type], listener ) !== -1,
                                      'Listener did not exist for type ' + type );
      
      this._events[type].splice( _.indexOf( this._events[type], listener ), 1 );
      
      if ( type in eventsRequiringBoundsValidation ) {
        this.changeBoundsEventCount( -1 );
        this._boundsEventSelfCount--;
      }
      
      // allow chaining
      return this;
    };
    
    /*
     * Fires an event to all event listeners attached to this node. It does not bubble down to
     * all ancestors with trails, like dispatchEvent does. Use fireEvent when you only want an event
     * that is relevant for a specific node, and ancestors don't need to be notified.
     */
    proto.fireEvent = function( type, args ) {
      assert && assert( _.contains( eventNames, type ),
                                      'unknown event type: ' + type );
      
      var events = this._events[type];
      var len = events.length;
      if ( len ) { // TODO: consider removing branch? is this even helpful?
        // TODO: reduce allocation? consider tmp array to hold these, or will that cause more memory usage?
        var copy = events.slice( 0 ); // defensive copy, in case listeners are added or removed as a side effect of a listener being called
        for ( var i = 0; i < len; i++ ) {
          copy[i]( args );
        }
      }
    };
  };
  var FixedNodeEvents = scenery.FixedNodeEvents;
  
  return FixedNodeEvents;
} );


