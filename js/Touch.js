// Copyright 2002-2012, University of Colorado

/**
 * Tracks a single touch point
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  
  var Finger = require( 'SCENERY/Finger' );
  
  var Touch = function( id, point, event ) {
    Finger.call( this );
    
    this.id = id;
    this.point = point;
    this.isTouch = true;
    this.trail = null;
  };
  
  Touch.prototype = _.extend( {}, Finger.prototype, {
    constructor: Touch,
    
    move: function( point, event ) {
      this.point = point;
    },
    
    end: function( point, event ) {
      this.point = point;
    },
    
    cancel: function( point, event ) {
      this.point = point;
    }
  } );
  
  return Touch;
} );
