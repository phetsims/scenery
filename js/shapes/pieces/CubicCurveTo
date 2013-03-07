// Copyright 2002-2012, University of Colorado

/**
 * Draws a cubic bezier curve
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );
  require( 'SCENERY/shapes/segments/Cubic' );
  
  Piece.CubicCurveTo = function( control1, control2, point ) {
    this.control1 = control1;
    this.control2 = control2;
    this.point = point;
  };
  Piece.CubicCurveTo.prototype = {
    constructor: Piece.CubicCurveTo,
    
    writeToContext: function( context ) {
      context.bezierCurveTo( this.control1.x, this.control1.y, this.control2.x, this.control2.y, this.point.x, this.point.y );
    },
    
    transformed: function( matrix ) {
      return [new Piece.CubicCurveTo( matrix.timesVector2( this.control1 ), matrix.timesVector2( this.control2 ), matrix.timesVector2( this.point ) )];
    },
    
    applyPiece: function( shape ) {
      // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-quadraticcurveto
      shape.ensure( this.controlPoint );
      var start = shape.getLastSubpath().getLastPoint();
      var cubic = new scenery.Segment.Cubic( start, this.control1, this.control2, this.point );
      shape.getLastSubpath().addSegment( cubic );
      shape.getLastSubpath().addPoint( this.point );
      if ( !cubic.invalid ) {
        shape.bounds = shape.bounds.union( cubic.bounds );
      }
    }
  };
  
  return Piece.CubicCurveTo;
} );
