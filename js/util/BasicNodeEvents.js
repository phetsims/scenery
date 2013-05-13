// Copyright 2002-2012, University of Colorado

/**
 * Mix-in for Node's event handling, in the original style
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.BasicNodeEvents = function( type ) {
    var proto = type.prototype;
    
    // this should be called in the constructor to initialize
    proto.initializeNodeEvents = function() {
      this._eventListeners = [];
    };
    
    proto.addEventListener = function( listener ) {
      // don't allow listeners to be added multiple times
      if ( _.indexOf( this._eventListeners, listener ) === -1 ) {
        this._eventListeners.push( listener );
      }
      return this;
    };
    
    proto.removeEventListener = function( listener ) {
      // ensure the listener is in our list
      assert && assert( _.indexOf( this._eventListeners, listener ) !== -1 );
      
      this._eventListeners.splice( _.indexOf( this._eventListeners, listener ), 1 );
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
      // TODO: performance: 8% bottleneck - consider storing separate locations for each event type
      var len = this._eventListeners.length;
      if ( len ) {
        var eventListenersCopy = this._eventListeners.slice( 0 );
        for ( var i = 0; i < len; i++ ) {
          var callback = eventListenersCopy[i][type];
          if ( callback ) {
            callback( args );
          }
        }
      }
    };
  };
  var BasicNodeEvents = scenery.BasicNodeEvents;
  
  return BasicNodeEvents;
} );


