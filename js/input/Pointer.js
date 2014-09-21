// Copyright 2002-2013, University of Colorado

/*
 * A pointer is an abstraction that includes a mouse and touch points (and possibly keys).
 *
 * TODO: add state tracking (dragging/panning/etc.) to pointer for convenience
 * TODO: consider an 'active' flag?
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  scenery.Pointer = function Pointer() {
    this.listeners = [];

    phetAllocation && phetAllocation( 'Pointer' );

    this.active = false; // whether a control is being actively manipulated by this pointer
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
    },

    // for mouse/touch/pen
    hasPointChanged: function( point ) {
      return this.point !== point && ( !point || !this.point || !this.point.equals( point ) );
    }
  };

  return Pointer;
} );
