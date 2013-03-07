// Copyright 2002-2012, University of Colorado

/**
 * Draws an arc.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Vector2 = require( 'DOT/Vector2' );
  
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );
  require( 'SCENERY/shapes/pieces/EllipticalArc' );
  require( 'SCENERY/shapes/segments/Line' );
  require( 'SCENERY/shapes/segments/Arc' );
  require( 'SCENERY/shapes/util/Subpath' );
  
  Piece.Arc = function( center, radius, startAngle, endAngle, anticlockwise ) {
    this.center = center;
    this.radius = radius;
    this.startAngle = startAngle;
    this.endAngle = endAngle;
    this.anticlockwise = anticlockwise;
  };
  Piece.Arc.prototype = {
    constructor: Piece.Arc,
    
    writeToContext: function( context ) {
      context.arc( this.center.x, this.center.y, this.radius, this.startAngle, this.endAngle, this.anticlockwise );
    },
    
    // TODO: test various transform types, especially rotations, scaling, shears, etc.
    transformed: function( matrix ) {
      // so we can handle reflections in the transform, we do the general case handling for start/end angles
      var startAngle = matrix.timesVector2( Vector2.createPolar( 1, this.startAngle ) ).minus( matrix.timesVector2( Vector2.ZERO ) ).angle();
      var endAngle = matrix.timesVector2( Vector2.createPolar( 1, this.endAngle ) ).minus( matrix.timesVector2( Vector2.ZERO ) ).angle();
      
      // reverse the 'clockwiseness' if our transform includes a reflection
      var anticlockwise = matrix.determinant() >= 0 ? this.anticlockwise : !this.anticlockwise;
      
      if ( matrix.scaling().x !== matrix.scaling().y ) {
        var radiusX = matrix.scaling().x * this.radius;
        var radiusY = matrix.scaling().y * this.radius;
        return [new Piece.EllipticalArc( matrix.timesVector2( this.center ), radiusX, radiusY, 0, startAngle, endAngle, anticlockwise )];
      } else {
        var radius = matrix.scaling().x * this.radius;
        return [new Piece.Arc( matrix.timesVector2( this.center ), radius, startAngle, endAngle, anticlockwise )];
      }
    },
    
    applyPiece: function( shape ) {
      // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-arc
      
      var arc = new scenery.Segment.Arc( this.center, this.radius, this.startAngle, this.endAngle, this.anticlockwise );
      
      // we are assuming that the normal conditions were already met (or exceptioned out) so that these actually work with canvas
      var startPoint = arc.start;
      var endPoint = arc.end;
      
      // if there is already a point on the subpath, and it is different than our starting point, draw a line between them
      if ( shape.hasSubpaths() && shape.getLastSubpath().getLength() > 0 && !startPoint.equals( shape.getLastSubpath().getLastPoint(), 0 ) ) {
        shape.getLastSubpath().addSegment( new scenery.Segment.Line( shape.getLastSubpath().getLastPoint(), startPoint ) );
      }
      
      if ( !shape.hasSubpaths() ) {
        shape.addSubpath( new scenery.Subpath() );
      }
      
      shape.getLastSubpath().addSegment( arc );
      
      // technically the Canvas spec says to add the start point, so we do this even though it is probably completely unnecessary (there is no conditional)
      shape.getLastSubpath().addPoint( startPoint );
      shape.getLastSubpath().addPoint( endPoint );
      
      // and update the bounds
      if ( !arc.invalid ) {
        shape.bounds = shape.bounds.union( arc.bounds );
      }
    }
  };
  
  return Piece.Arc;
} );
