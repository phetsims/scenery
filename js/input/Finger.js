// Copyright 2002-2012, University of Colorado

/*
 * Conventionally set flags on a finger: TODO: add this state tracking to finger for convenience
 * dragging - whether the finger is dragging something
 *
 * TODO: consider an 'active' flag?
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
