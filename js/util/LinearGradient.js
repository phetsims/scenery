// Copyright 2002-2012, University of Colorado

/**
 * A linear gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
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
    getCanvasFill: function( context ) {
      var gradient = context.createLinearGradient( this.start.x, this.start.y, this.end.x, this.end.y );
      _.each( this.stops, function( stop ) {
        gradient.addColorStop( stop.ratio, stop.color );
      } );
    },
    
    // seems we need the defs: http://stackoverflow.com/questions/7614209/linear-gradients-in-svg-without-defs
    // SVG: spreadMethod 'pad' 'reflect' 'repeat' - find Canvas usage
    getSVGDefinition: function( id ) {
      /* Approximate example of what we are creating:
      <linearGradient id="grad2" x1="0" y1="0" x2="100" y2="0" gradientUnits="userSpaceOnUse">
        <stop offset="0" style="stop-color:rgb(255,255,0);stop-opacity:1" />
        <stop offset="0.5" style="stop-color:rgba(255,255,0,0);stop-opacity:0" />
        <stop offset="1" style="stop-color:rgb(255,0,0);stop-opacity:1" />
      </linearGradient>
      */
      var svgns = 'http://www.w3.org/2000/svg'; // TODO: store this in a common place!
      var definition = document.createElementNS( svgns, 'linearGradient' );
      definition.setAttribute( 'id', id );
      definition.setAttribute( 'gradientUnits', 'userSpaceOnUse' ); // so we don't depend on the bounds of the object being drawn with the gradient
      definition.setAttribute( 'x0', this.start.x );
      definition.setAttribute( 'y0', this.start.y );
      definition.setAttribute( 'x1', this.end.x );
      definition.setAttribute( 'y1', this.end.y );
      
      _.each( this.stops, function( stop ) {
        var stopElement = document.createElementNS( svgns, 'stop' );
        stopElement.setAttribute( 'offset', stop.ratio );
        stopElement.setAttribute( 'style', 'stop-color: ' + stop.color.withAlpha( 1 ).getCSS() + '; stop-opacity: ' + stop.color.a.toFixed( 20 ) + ';' );
        definition.appendChild( stopElement );
      } );
      
      return definition;
    }
  };
  
  return LinearGradient;
} );
