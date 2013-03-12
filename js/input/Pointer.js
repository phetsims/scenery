// Copyright 2002-2012, University of Colorado

/*
 * A pointer is an abstraction that includes a mouse and touch points (and possibly keys).
 *
 * TODO: add state tracking (dragging/panning/etc.) to pointer for convenience
 * TODO: consider an 'active' flag?
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Pointer = function() {
    this.listeners = [];
  };
  var Pointer = scenery.Pointer;
  
  Pointer.prototype = {
    constructor: Pointer,
    
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
  
  return Pointer;
} );
