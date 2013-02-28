// Copyright 2002-2012, University of Colorado

/*
 * A finger is an abstraction that includes a mouse and touch points (and possibly keys).
 *
 * TODO: add state tracking (dragging/panning/etc.) to finger for convenience
 * TODO: consider an 'active' flag?
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
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
  
  return Finger;
} );
