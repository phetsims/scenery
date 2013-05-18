// Copyright 2002-2012, University of Colorado

/**
 * Tracks a stylus ('pen') or something with tilt and pressure information
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  var Pointer = require( 'SCENERY/input/Pointer' ); // extends Pointer
  
  scenery.Pen = function Pen( id, point, event ) {
    Pointer.call( this );
    
    this.id = id;
    this.point = point;
    this.isPen = true;
    this.trail = null;
    
    this.type = 'pen';
  };
  var Pen = scenery.Pen;
  
  inherit( Pen, Pointer, {
    move: function( point, event ) {
      this.point = point;
    },
    
    end: function( point, event ) {
      this.point = point;
    },
    
    cancel: function( point, event ) {
      this.point = point;
    },
    
    toString: function() {
      return 'Pen#' + this.id;
    }
  } );
  
  return Pen;
} );
