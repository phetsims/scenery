// Copyright 2002-2012, University of Colorado

/**
 * Tracks the mouse state
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Pointer = require( 'SCENERY/input/Pointer' ); // inherits from Pointer
  
  scenery.Mouse = function Mouse() {
    Pointer.call( this );
    
    this.point = null;
    
    this.leftDown = false;
    this.middleDown = false;
    this.rightDown = false;
    
    this.isMouse = true;
    
    this.trail = null;
    
    this.type = 'mouse';
  };
  var Mouse = scenery.Mouse;
  
  Mouse.prototype = _.extend( {}, Pointer.prototype, {
    constructor: Mouse,
    
    down: function( point, event ) {
      this.point = point;
      switch( event.button ) {
        case 0: this.leftDown = true; break;
        case 1: this.middleDown = true; break;
        case 2: this.rightDown = true; break;
      }
    },
    
    up: function( point, event ) {
      this.point = point;
      switch( event.button ) {
        case 0: this.leftDown = false; break;
        case 1: this.middleDown = false; break;
        case 2: this.rightDown = false; break;
      }
    },
    
    move: function( point, event ) {
      this.point = point;
    },
    
    over: function( point, event ) {
      this.point = point;
    },
    
    out: function( point, event ) {
      // TODO: how to handle the mouse out-of-bounds
      this.point = null;
    },
    
    toString: function() {
      return 'Mouse';
    }
  } );
  
  return Mouse;
} );
