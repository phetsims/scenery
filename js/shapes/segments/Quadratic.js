// Copyright 2002-2012, University of Colorado

/**
 * Quadratic Bezier segment
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );

  var scenery = require( 'SCENERY/scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );

  var Segment = require( 'SCENERY/shapes/segments/Segment' );
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );

  Segment.Quadratic = function( start, control, end, skipComputations ) {
    this.start = start;
    this.control = control;
    this.end = end;
    
    if ( start.equals( end, 0 ) && start.equals( control, 0 ) ) {
      this.invalid = true;
      return;
    }
    
    var t;
    
    // allows us to skip unnecessary computation in the subdivision steps
    if ( skipComputations ) {
      return;
    }
    
    var controlIsStart = start.equals( control );
    var controlIsEnd = end.equals( control );
    // ensure the points are distinct
    assert && assert( !controlIsStart || !controlIsEnd );
    
    // allow either the start or end point to be the same as the control point (necessary if you do a quadraticCurveTo on an empty path)
    // tangents go through the control point, which simplifies things
    this.startTangent = controlIsStart ? end.minus( start ).normalized() : control.minus( start ).normalized();
    this.endTangent = controlIsEnd ? end.minus( start ).normalized() : end.minus( control ).normalized();
    
    // calculate our temporary guaranteed lower bounds based on the end points
    this.bounds = new Bounds2( Math.min( start.x, end.x ), Math.min( start.y, end.y ), Math.max( start.x, end.x ), Math.max( start.y, end.y ) );
    
    // compute x and y where the derivative is 0, so we can include this in the bounds
    var divisorX = 2 * ( end.x - 2 * control.x + start.x );
    if ( divisorX !== 0 ) {
      t = -2 * ( control.x - start.x ) / divisorX;
      
      if ( t > 0 && t < 1 ) {
        this.bounds = this.bounds.withPoint( this.positionAt( t ) );
      }
    }
    var divisorY = 2 * ( end.y - 2 * control.y + start.y );
    if ( divisorY !== 0 ) {
      t = -2 * ( control.y - start.y ) / divisorY;
      
      if ( t > 0 && t < 1 ) {
        this.bounds = this.bounds.withPoint( this.positionAt( t ) );
      }
    }
  };
  Segment.Quadratic.prototype = {
    constructor: Segment.Quadratic,
    
    // can be described from t=[0,1] as: (1-t)^2 start + 2(1-t)t control + t^2 end
    positionAt: function( t ) {
      var mt = 1 - t;
      return this.start.times( mt * mt ).plus( this.control.times( 2 * mt * t ) ).plus( this.end.times( t * t ) );
    },
    
    // derivative: 2(1-t)( control - start ) + 2t( end - control )
    gradientAt: function( t ) {
      return this.control.minus( this.start ).times( 2 * ( 1 - t ) ).plus( this.end.minus( this.control ).times( 2 * t ) );
    },
    
    offsetTo: function( r, reverse ) {
      // TODO: implement more accurate method at http://www.antigrain.com/research/adaptive_bezier/index.html
      // TODO: or more recently (and relevantly): http://www.cis.usouthal.edu/~hain/general/Publications/Bezier/BezierFlattening.pdf
      var curves = [this];
      
      // subdivide this curve
      var depth = 5; // generates 2^depth curves
      for ( var i = 0; i < depth; i++ ) {
        curves = _.flatten( _.map( curves, function( curve ) {
          return curve.subdivided( true );
        } ));
      }
      
      var offsetCurves = _.map( curves, function( curve ) { return curve.approximateOffset( r ); } );
      
      if ( reverse ) {
        offsetCurves.reverse();
        offsetCurves = _.map( offsetCurves, function( curve ) { return curve.reversed( true ); } );
      }
      
      var result = _.map( offsetCurves, function( curve ) {
        return new Piece.QuadraticCurveTo( curve.control, curve.end );
      } );
      
      return result;
    },
    
    subdivided: function( skipComputations ) {
      // de Casteljau method
      var leftMid = this.start.plus( this.control ).times( 0.5 );
      var rightMid = this.control.plus( this.end ).times( 0.5 );
      var mid = leftMid.plus( rightMid ).times( 0.5 );
      return [
        new Segment.Quadratic( this.start, leftMid, mid, skipComputations ),
        new Segment.Quadratic( mid, rightMid, this.end, skipComputations )
      ];
    },
    
    reversed: function( skipComputations ) {
      return new Segment.Quadratic( this.end, this.control, this.start );
    },
    
    approximateOffset: function( r ) {
      return new Segment.Quadratic(
        this.start.plus( ( this.start.equals( this.control ) ? this.end.minus( this.start ) : this.control.minus( this.start ) ).perpendicular().normalized().times( r ) ),
        this.control.plus( this.end.minus( this.start ).perpendicular().normalized().times( r ) ),
        this.end.plus( ( this.end.equals( this.control ) ? this.end.minus( this.start ) : this.end.minus( this.control ) ).perpendicular().normalized().times( r ) )
      );
    },
    
    toPieces: function() {
      return [ new Piece.QuadraticCurveTo( this.control, this.end ) ];
    },
    
    getSVGPathFragment: function() {
      return 'Q ' + this.control.x + ' ' + this.control.y + ' ' + this.end.x + ' ' + this.end.y;
    },
    
    strokeLeft: function( lineWidth ) {
      return this.offsetTo( -lineWidth / 2, false );
    },
    
    strokeRight: function( lineWidth ) {
      return this.offsetTo( lineWidth / 2, true );
    },
    
    intersectsBounds: function( bounds ) {
      throw new Error( 'Segment.Quadratic.intersectsBounds unimplemented' ); // TODO: implement
    },
    
    // returns the resultant winding number of this ray intersecting this segment.
    windingIntersection: function( ray ) {
      // TODO: optimization
      
      // find the rotation that will put our ray in the direction of the x-axis so we can only solve for y=0 for intersections
      var inverseMatrix = Matrix3.rotation2( -ray.dir.angle() );
      assert && assert( inverseMatrix.timesVector2( ray.dir ).x > 0.99 ); // verify that we transform the unit vector to the x-unit
      
      var p0 = inverseMatrix.timesVector2( this.start );
      var p1 = inverseMatrix.timesVector2( this.control );
      var p2 = inverseMatrix.timesVector2( this.end );
      
      var det = p1.y * p1.y - p0.y * p2.y;
      if ( det < 0.00000001 ) {
        return 0; // no intersection with the mathematical (extended) curve
      }
      
      // the two t values, which should be valid in our regular coordinate system
      var ta = ( p0.y - p1.y + Math.sqrt( det ) ) / ( p0.y - 2 * p1.y + p2.y );
      var tb = ( p0.y - p1.y - Math.sqrt( det ) ) / ( p0.y - 2 * p1.y + p2.y );
      
      var da = this.positionAt( ta ).minus( ray.pos );
      var db = this.positionAt( tb ).minus( ray.pos );
      
      var aValid = ta > 0 && da.dot( ray.dir ) > 0;
      var bValid = tb > 0 && db.dot( ray.dir ) > 0;
      
      var result = 0;
      
      if ( aValid ) {
        result += ray.dir.perpendicular().dot( this.gradientAt( ta ) ) < 0 ? 1 : -1;
      }
      
      if ( bValid ) {
        result += ray.dir.perpendicular().dot( this.gradientAt( tb ) ) < 0 ? 1 : -1;
      }
      
      return result;
    }
  };
  
  return Segment.Quadratic;
} );
