// Copyright 2002-2012, University of Colorado

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
    'bounds'
  ];
  
  scenery.SplitNodeEvents = function( type ) {
    var proto = type.prototype;
    
    // this should be called in the constructor to initialize
    proto.initializeNodeEvents = function() {
      this._eventListeners = [];
      this._events = {};
      
      var node = this;
      // TODO: performance: consider delaying this? Could affect memory usage?
      _.each( eventNames, function( name ) {
        node._events[name] = [];
      } );
    };
    
    proto.addEventListener = function( listener ) {
      var node = this;
      // don't allow listeners to be added multiple times
      if ( _.indexOf( this._eventListeners, listener ) === -1 ) {
        this._eventListeners.push( listener );
        
        _.each( eventNames, function( name ) {
          if ( listener[name] ) {
            node._events[name].push( listener[name] );
          }
        } );
      }
      return this;
    };
    
    proto.removeEventListener = function( listener ) {
      var node = this;
      // ensure the listener is in our list
      sceneryAssert && sceneryAssert( _.indexOf( this._eventListeners, listener ) !== -1 );
      
      this._eventListeners.splice( _.indexOf( this._eventListeners, listener ), 1 );
      
      _.each( eventNames, function( name ) {
        if ( listener[name] ) {
          var arr = node._events[name];
          sceneryAssert && sceneryAssert( _.indexOf( arr, listener[name] ) !== -1 );
          arr.splice( _.indexOf( arr, listener[name] ), 1 );
        }
      } );
      return this;
    };
    
    proto.getEventListeners = function() {
      return this._eventListeners.slice( 0 ); // defensive copy
    };
    
    /*
     * Fires an event to all event listeners attached to this node. It does not bubble down to
     * all ancestors with trails, like dispatchEvent does. Use fireEvent when you only want an event
     * that is relevant for a specific node, and ancestors don't need to be notified.
     */
    proto.fireEvent = function( type, args ) {
      sceneryAssert && sceneryAssert( _.contains( eventNames, type ), 'unknown event type: ' + type );
      
      var array = this._events[type];
      var len = array.length;
      if ( len ) { // TODO: consider removing branch? is this even helpful?
        var copy = array.slice( 0 );
        for ( var i = 0; i < len; i++ ) {
          copy[i]( args );
        }
      }
    };
  };
  var SplitNodeEvents = scenery.SplitNodeEvents;
  
  return SplitNodeEvents;
} );


