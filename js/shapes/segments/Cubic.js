// Copyright 2002-2012, University of Colorado

/**
 * Cubic Bezier segment.
 *
 * See http://www.cis.usouthal.edu/~hain/general/Publications/Bezier/BezierFlattening.pdf for info
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );

  var scenery = require( 'SCENERY/scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var Segment = require( 'SCENERY/shapes/segments/Segment' );
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );

  Segment.Cubic = function( start, control1, control2, end ) {
    this.start = start;
    this.control1 = control1;
    this.control2 = control2;
    this.end = end;
    
    this.startTangent = this.gradientAt( 0 ).normalized();
    this.endTangent = this.gradientAt( 1 ).normalized();
    
    if ( start.equals( end, 0 ) && start.equals( control1, 0 ) && start.equals( control2, 0 ) ) {
      this.invalid = true;
      return;
    }
    
    this.bounds = Bounds2.NOTHING;
    this.bounds = this.bounds.withPoint( this.start );
    this.bounds = this.bounds.withPoint( this.end );
    
  };
  Segment.Cubic.prototype = {
    // position: (1 - t)^3*start + 3*(1 - t)^2*t*control1 + 3*(1 - t) t^2*control2 + t^3*end
    positionAt: function( t ) {
      var mt = 1 - t;
      return this.start.times( mt * mt * mt ).plus( this.control1.times( 3 * mt * mt * t ) ).plus( this.control2.times( 3 * mt * t * t ) ).plus( this.end.times( t * t * t ) );
    },
    
    // derivative: -3 p0 (1 - t)^2 + 3 p1 (1 - t)^2 - 6 p1 (1 - t) t + 6 p2 (1 - t) t - 3 p2 t^2 + 3 p3 t^2
    gradientAt: function( t ) {
      var mt = 1 - t;
      return this.start.times( -3 * mt * mt ).plus( this.control1.times( 3 * mt * mt - 6 * mt * t ) ).plus( this.control2.times( 6 * mt * t - 3 * t * t ) ).plus( this.end.times( t * t ) );
    }
  };
  
  return Segment.Cubic;
} );
