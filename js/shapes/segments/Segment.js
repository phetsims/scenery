// Copyright 2002-2012, University of Colorado

/**
 * A segment represents a specific curve with a start and end.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );

  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Segment = {
    /*
     * Will contain (for segments):
     * properties:
     * start        - start point of this segment
     * end          - end point of this segment
     * startTangent - the tangent vector (normalized) to the segment at the start, pointing in the direction of motion (from start to end)
     * endTangent   - the tangent vector (normalized) to the segment at the end, pointing in the direction of motion (from start to end)
     * bounds       - the bounding box for the segment
     *
     * methods:
     * positionAt( t )          - returns the position parametrically, with 0 <= t <= 1. this does NOT guarantee a constant magnitude tangent... don't feel like adding elliptical functions yet!
     * tangentAt( t )           - returns the tangent (dx/dt, dy/dt) parametrically, with 0 <= t <= 1.
     * toPieces                 - returns an array of pieces that are equivalent to this segment, assuming start points are preserved
     *                              TODO: is toPieces that valuable? it doesn't seem to have a strict guarantee on checking what the last segment did right now
     * getSVGPathFragment       - returns a string containing the SVG path. assumes that the start point is already provided, so anything that calls this needs to put the M calls first
     * strokeLeft( lineWidth )  - returns an array of pieces that will draw an offset curve on the logical left side
     * strokeRight( lineWidth ) - returns an array of pieces that will draw an offset curve on the logical right side
     * intersectsBounds         - whether this segment intersects the specified bounding box (not just the segment's bounding box, but the actual segment)
     * windingIntersection      - returns the winding number for intersection with a ray
     */
  };
  var Segment = scenery.Segment;
  
  return Segment;
} );
