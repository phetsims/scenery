// Copyright 2002-2012, University of Colorado

/**
 * A radial gradient that can be passed into the 'fill' or 'stroke' parameters.
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
  
  // TODO: support Vector2s for p0 and p1
  scenery.RadialGradient = function( x0, y0, r0, x1, y1, r1 ) {
    this.start = new Vector2( x0, y0 );
    this.end = new Vector2( x1, y1 );
    this.startRadius = r0;
    this.endRadius = r1;
    
    // linear function from radius to point on the line from start to end
    this.focalPoint = this.start.plus( this.end.minus( this.start ).times( this.startRadius / ( this.startRadius - this.endRadius ) ) );
    
    // make sure that the focal point is in both circles. SVG doesn't support rendering outside of them
    if ( this.startRadius >= this.endRadius ) {
      assert && assert( this.focalPoint.minus( this.start ).magnitude() <= this.startRadius );
    } else {
      assert && assert( this.focalPoint.minus( this.end ).magnitude() <= this.endRadius );
    }
    
    this.stops = [];
    this.lastStopRatio = 0;
    
    // TODO: make a global spot that will have a 'useless' context for these purposes?
    this.canvasGradient = document.createElement( 'canvas' ).getContext( '2d' ).createRadialGradient( x0, y0, r0, x1, y1, r1 );
    
    this.transformMatrix = null;
  };
  var RadialGradient = scenery.RadialGradient;
  
  RadialGradient.prototype = {
    constructor: RadialGradient,
    
    addColorStop: function( ratio, color ) {
      if ( this.lastStopRatio > ratio ) {
        // fail out, since browser quirks go crazy for this case
        throw new Error( 'Color stops not specified in the order of increasing ratios' );
      } else {
        this.lastStopRatio = ratio;
      }
      
      this.stops.push( { ratio: ratio, color: color } );
      this.canvasGradient.addColorStop( ratio, color );
      return this;
    },
    
    setTransformMatrix: function( transformMatrix ) {
      this.transformMatrix = transformMatrix;
      return this;
    },
    
    getCanvasStyle: function() {
      return this.canvasGradient;
    },
    
    getSVGDefinition: function( id ) {
      var startIsLarger = this.startRadius > this.endRadius;
      var largePoint = startIsLarger ? this.start : this.end;
      var smallPoint = startIsLarger ? this.end : this.start;
      var maxRadius = Math.max( this.startRadius, this.endRadius );
      var minRadius = Math.min( this.startRadius, this.endRadius );
      
      var svgns = 'http://www.w3.org/2000/svg'; // TODO: store this in a common place!
      var definition = document.createElementNS( svgns, 'radialGradient' );
      
      // TODO:
      definition.setAttribute( 'id', id );
      definition.setAttribute( 'gradientUnits', 'userSpaceOnUse' ); // so we don't depend on the bounds of the object being drawn with the gradient
      definition.setAttribute( 'cx', largePoint.x );
      definition.setAttribute( 'cy', largePoint.y );
      definition.setAttribute( 'r', maxRadius );
      definition.setAttribute( 'fx', this.focalPoint.x );
      definition.setAttribute( 'fy', this.focalPoint.y );
      if ( this.transformMatrix ) {
        definition.setAttribute( 'gradientTransform', this.transformMatrix.getSVGTransform() );
      }
      
      // maps x linearly from [a0,b0] => [a1,b1]
      function linearMap( a0, b0, a1, b1, x ) {
        return a1 + ( x - a0 ) * ( b1 - a1 ) / ( b0 - a0 );
      }
      
      function applyStop( stop ) {
        // flip the stops if the start has a larger radius
        var ratio = startIsLarger ? 1 - stop.ratio : stop.ratio;
        
        // scale the stops properly if the smaller radius isn't 0
        if ( minRadius > 0 ) {
          // scales our ratio from [0,1] => [minRadius/maxRadius,0]
          ratio = linearMap( 0, 1, minRadius / maxRadius, 1, ratio );
        }
        
        // TODO: store color in our stops array, so we don't have to create additional objects every time?
        var color = new scenery.Color( stop.color );
        var stopElement = document.createElementNS( svgns, 'stop' );
        stopElement.setAttribute( 'offset', ratio );
        stopElement.setAttribute( 'style', 'stop-color: ' + color.withAlpha( 1 ).getCSS() + '; stop-opacity: ' + color.a.toFixed( 20 ) + ';' );
        definition.appendChild( stopElement );
      }
      
      var i;
      // switch the direction we apply stops in, so that the ratios always are increasing.
      if ( startIsLarger ) {
        for ( i = this.stops.length - 1; i >= 0; i-- ) {
          applyStop( this.stops[i] );
        }
      } else {
        for ( i = 0; i < this.stops.length; i++ ) {
          applyStop( this.stops[i] );
        }
      }
      
      return definition;
    },
    
    toString: function() {
      var result = 'new scenery.RadialGradient( ' + this.start.x + ', ' + this.start.y + ', ' + this.startRadius + ', ' + this.end.x + ', ' + this.end.y + ', ' + this.endRadius + ' )';
      
      _.each( this.stops, function( stop ) {
        result += '.addColorStop( ' + stop.ratio + ', \'' + stop.color + '\' )';
      } );
      
      return result;
    }
  };
  
  return RadialGradient;
} );
