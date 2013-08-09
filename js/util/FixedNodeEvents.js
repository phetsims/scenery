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
    'bounds',
    'resize',
    'boundsAccuracy'
  ];
  
  scenery.FixedNodeEvents = function FixedNodeEvents( type ) {
    var proto = type.prototype;
    
    // this should be called in the constructor to initialize
    proto.initializeNodeEvents = function() {
      this._events = {};
      
      var node = this;
      
      // TODO: performance: consider delaying this? Could affect memory usage?
      _.each( eventNames, function( name ) {
        node._events[name] = [];
      } );
    };
    
    /**
     * @param {String}   type     The type of event, like 'resize' or 'bounds'
     * @param {Function} listener Callback, called with arguments that depend on the event type
     */
    proto.addEventListener = function( type, listener ) {
      sceneryAssert && sceneryAssert( type !== undefined && listener !== undefined,
                                      'Both a type and listener are required for addEventListener' );
      
      // most commonly a bug, maybe there will be a good use case? can always work around by wrapping with a new function each time
      sceneryAssert && sceneryAssert( _.indexOf( this._events[type], listener ),
                                      'Event listener was already there for addEventListener with type ' + type );
      
      this._events[type].push( listener );
      
      // allow chaining
      return this;
    };
    
    /**
     * @param {String}   type     The type of event, like 'resize' or 'bounds'
     * @param {Function} listener The callback to remove.
     */
    proto.removeEventListener = function( type, listener ) {
      sceneryAssert && sceneryAssert( type !== undefined && listener !== undefined,
                                      'Both a type and listener are required for removeEventListener' );
      
      // ensure the listener is in our list
      sceneryAssert && sceneryAssert( _.indexOf( this._events[type], listener ) !== -1,
                                      'Listener did not exist for type ' + type );
      
      this._events[type].splice( _.indexOf( this._events[type], listener ), 1 );
      
      // allow chaining
      return this;
    };
    
    /*
     * Fires an event to all event listeners attached to this node. It does not bubble down to
     * all ancestors with trails, like dispatchEvent does. Use fireEvent when you only want an event
     * that is relevant for a specific node, and ancestors don't need to be notified.
     */
    proto.fireEvent = function( type, args ) {
      sceneryAssert && sceneryAssert( _.contains( eventNames, type ),
                                      'unknown event type: ' + type );
      
      var events = this._events[type];
      var len = events.length;
      if ( len ) { // TODO: consider removing branch? is this even helpful?
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


