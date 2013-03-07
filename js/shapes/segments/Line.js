// Copyright 2002-2012, University of Colorado

/**
 * Linear segment
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );

  var scenery = require( 'SCENERY/scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var lineLineIntersection = require( 'DOT/Util' ).lineLineIntersection;
  
  var Segment = require( 'SCENERY/shapes/segments/Segment' );
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );

  Segment.Line = function( start, end ) {
    this.start = start;
    this.end = end;
    
    if ( start.equals( end, 0 ) ) {
      this.invalid = true;
      return;
    }
    
    this.startTangent = end.minus( start ).normalized();
    this.endTangent = this.startTangent;
    
    // acceleration for intersection
    this.bounds = new Bounds2().withPoint( start ).withPoint( end );
  };
  Segment.Line.prototype = {
    constructor: Segment.Line,
    
    toPieces: function() {
      return [ new Piece.LineTo( this.end ) ];
    },
    
    getSVGPathFragment: function() {
      return 'L ' + this.end.x + ' ' + this.end.y;
    },
    
    strokeLeft: function( lineWidth ) {
      return [ new Piece.LineTo( this.end.plus( this.endTangent.perpendicular().negated().times( lineWidth / 2 ) ) ) ];
    },
    
    strokeRight: function( lineWidth ) {
      return [ new Piece.LineTo( this.start.plus( this.startTangent.perpendicular().times( lineWidth / 2 ) ) ) ];
    },
    
    intersectsBounds: function( bounds ) {
      throw new Error( 'Segment.Line.intersectsBounds unimplemented' ); // TODO: implement
    },
    
    // returns the resultant winding number of this ray intersecting this segment.
    windingIntersection: function( ray ) {
      var start = this.start;
      var end = this.end;
      
      var intersection = lineLineIntersection( start, end, ray.pos, ray.pos.plus( ray.dir ) );
      
      if ( !isFinite( intersection.x ) || !isFinite( intersection.y ) ) {
        // lines must be parallel
        return 0;
      }
      
      // check to make sure our point is in our line segment (specifically, in the bounds (start,end], not including the start point so we don't double-count intersections)
      if ( start.x !== end.x && ( start.x > end.x ? ( intersection.x >= start.x || intersection.x < end.x ) : ( intersection.x <= start.x || intersection.x > end.x ) ) ) {
        return 0;
      }
      if ( start.y !== end.y && ( start.y > end.y ? ( intersection.y >= start.y || intersection.y < end.y ) : ( intersection.y <= start.y || intersection.y > end.y ) ) ) {
        return 0;
      }
      
      // make sure the intersection is not behind the ray
      var t = intersection.minus( ray.pos ).dot( ray.dir );
      if ( t < 0 ) {
        return 0;
      }
      
      // return the proper winding direction depending on what way our line intersection is "pointed"
      return ray.dir.perpendicular().dot( end.minus( start ) ) < 0 ? 1 : -1;
    }
  };
  
  return Segment.Line;
} );
