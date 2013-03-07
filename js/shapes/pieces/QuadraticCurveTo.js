// Copyright 2002-2012, University of Colorado

/**
 * Draws a quadratic bezier curve
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );
  require( 'SCENERY/shapes/segments/Quadratic' );
  
  Piece.QuadraticCurveTo = function( controlPoint, point ) {
    this.controlPoint = controlPoint;
    this.point = point;
  };
  Piece.QuadraticCurveTo.prototype = {
    constructor: Piece.QuadraticCurveTo,
    
    writeToContext: function( context ) {
      context.quadraticCurveTo( this.controlPoint.x, this.controlPoint.y, this.point.x, this.point.y );
    },
    
    transformed: function( matrix ) {
      return [new Piece.QuadraticCurveTo( matrix.timesVector2( this.controlPoint ), matrix.timesVector2( this.point ) )];
    },
    
    applyPiece: function( shape ) {
      // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-quadraticcurveto
      shape.ensure( this.controlPoint );
      var start = shape.getLastSubpath().getLastPoint();
      var quadratic = new scenery.Segment.Quadratic( start, this.controlPoint, this.point );
      shape.getLastSubpath().addSegment( quadratic );
      shape.getLastSubpath().addPoint( this.point );
      if ( !quadratic.invalid ) {
        shape.bounds = shape.bounds.union( quadratic.bounds );
      }
    }
  };
  
  return Piece.QuadraticCurveTo;
} );
