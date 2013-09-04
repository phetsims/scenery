// Copyright 2002-2013, University of Colorado

/**
 * Tracks a single touch point
 *
 * IE guidelines for Touch-friendly sites: http://blogs.msdn.com/b/ie/archive/2012/04/20/guidelines-for-building-touch-friendly-sites.aspx
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  var Pointer = require( 'SCENERY/input/Pointer' ); // extends Pointer
  
  scenery.Touch = function Touch( id, point, event ) {
    Pointer.call( this );
    
    this.id = id;
    this.point = point;
    this.isTouch = true;
    this.trail = null;
    
    this.type = 'touch';
  };
  var Touch = scenery.Touch;
  
  inherit( Pointer, Touch, {
    move: function( point, event ) {
      // if ( this.point ) { this.point.freeToPool(); }
      this.point = point;
    },
    
    end: function( point, event ) {
      // if ( this.point ) { this.point.freeToPool(); }
      this.point = point;
    },
    
    cancel: function( point, event ) {
      // if ( this.point ) { this.point.freeToPool(); }
      this.point = point;
    },
    
    toString: function() {
      return 'Touch#' + this.id;
    }
  } );
  
  return Touch;
} );
