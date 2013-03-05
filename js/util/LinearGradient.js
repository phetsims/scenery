// Copyright 2002-2012, University of Colorado

/**
 * A linear gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Vector2 = require( 'DOT/Vector2' );

  // TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
  scenery.LinearGradient = function( x0, y0, x1, y1 ) {
    var usesVectors = y1 === undefined;
    if ( usesVectors ) {
      assert && assert( ( x0 instanceof Vector2 ) && ( y0 instanceof Vector2 ), 'If less than 4 parameters are given, the first two parameters must be Vector2' );
    }
    this.start = usesVectors ? x0 : new Vector2( x0, y0 );
    this.end = usesVectors ? y0 : new Vector2( x1, y1 );
    
    this.stops = [];
  };
  var LinearGradient = scenery.LinearGradient;
  
  LinearGradient.prototype = {
    constructor: LinearGradient,
    
    addColorStop: function( ratio, color ) {
      this.stops.push( { ratio: ratio, color: color } );
    },
    
    // TODO: for performance, we should create a Canvas 'gradient' and keep it persistently
    getCanvasGradient: function( context ) {
      var gradient = context.createLinearGradient( this.start.x, this.start.y, this.end.x, this.end.y );
      _.each( this.stops, function( stop ) {
        gradient.addColorStop( stop.ratio, stop.color );
      } );
    }
  };
  
  return LinearGradient;
} );
