// Copyright 2002-2012, University of Colorado

/**
 * Arc segment
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );

  var scenery = require( 'SCENERY/scenery' );
  
  var Vector2 = require( 'DOT/Vector2' );
  var Bounds2 = require( 'DOT/Bounds2' );

  var Segment = require( 'SCENERY/shapes/segments/Segment' );
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );

  Segment.Arc = function( center, radius, startAngle, endAngle, anticlockwise ) {
    this.center = center;
    this.radius = radius;
    this.startAngle = startAngle;
    this.endAngle = endAngle;
    this.anticlockwise = anticlockwise;
    
    this.start = this.positionAtAngle( startAngle );
    this.end = this.positionAtAngle( endAngle );
    this.startTangent = this.tangentAtAngle( startAngle );
    this.endTangent = this.tangentAtAngle( endAngle );
    
    if ( radius <= 0 || startAngle === endAngle ) {
      this.invalid = true;
      return;
    }
    // constraints
    assert && assert( !( ( !anticlockwise && endAngle - startAngle <= -Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle <= -Math.PI * 2 ) ), 'Not handling arcs with start/end angles that show differences in-between browser handling' );
    assert && assert( !( ( !anticlockwise && endAngle - startAngle > Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle > Math.PI * 2 ) ), 'Not handling arcs with start/end angles that show differences in-between browser handling' );
    
    var isFullPerimeter = ( !anticlockwise && endAngle - startAngle >= Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle >= Math.PI * 2 );
    
    // compute an angle difference that represents how "much" of the circle our arc covers
    this.angleDifference = this.anticlockwise ? this.startAngle - this.endAngle : this.endAngle - this.startAngle;
    if ( this.angleDifference < 0 ) {
      this.angleDifference += Math.PI * 2;
    }
    assert && assert( this.angleDifference >= 0 ); // now it should always be zero or positive
    
    // acceleration for intersection
    this.bounds = Bounds2.NOTHING;
    this.bounds = this.bounds.withPoint( this.start );
    this.bounds = this.bounds.withPoint( this.end );
    
    // for bounds computations
    var that = this;
    function boundsAtAngle( angle ) {
      if ( that.containsAngle( angle ) ) {
        // the boundary point is in the arc
        that.bounds = that.bounds.withPoint( center.plus( Vector2.createPolar( radius, angle ) ) );
      }
    }
    
    // if the angles are different, check extrema points
    if ( startAngle !== endAngle ) {
      // check all of the extrema points
      boundsAtAngle( 0 );
      boundsAtAngle( Math.PI / 2 );
      boundsAtAngle( Math.PI );
      boundsAtAngle( 3 * Math.PI / 2 );
    }
  };
  Segment.Arc.prototype = {
    constructor: Segment.Arc,
    
    angleAt: function( t ) {
      return this.startAngle + ( this.endAngle - this.startAngle ) * t;
    },
    
    positionAt: function( t ) {
      return this.positionAtAngle( this.angleAt( t ) );
    },
    
    tangentAt: function( t ) {
      return this.tangentAtAngle( this.angleAt( t ) );
    },
    
    positionAtAngle: function( angle ) {
      return this.center.plus( Vector2.createPolar( this.radius, angle ) );
    },
    
    tangentAtAngle: function( angle ) {
      return Vector2.createPolar( 1, angle + this.anticlockwise ? Math.PI / 2 : -Math.PI / 2 );
    },
    
    // TODO: refactor? shared with Segment.EllipticalArc
    containsAngle: function( angle ) {
      // transform the angle into the appropriate coordinate form
      // TODO: check anticlockwise version!
      var normalizedAngle = this.anticlockwise ? angle - this.endAngle : angle - this.startAngle;
      
      // get the angle between 0 and 2pi
      var positiveMinAngle = normalizedAngle % ( Math.PI * 2 );
      // check this because modular arithmetic with negative numbers reveal a negative number
      if ( positiveMinAngle < 0 ) {
        positiveMinAngle += Math.PI * 2;
      }
      
      return positiveMinAngle <= this.angleDifference;
    },
    
    toPieces: function() {
      return [ new Piece.Arc( this.center, this.radius, this.startAngle, this.endAngle, this.anticlockwise ) ];
    },
    
    getSVGPathFragment: function() {
      // see http://www.w3.org/TR/SVG/paths.html#PathDataEllipticalArcCommands for more info
      // rx ry x-axis-rotation large-arc-flag sweep-flag x y
      
      var epsilon = 0.01; // allow some leeway to render things as 'almost circles'
      var sweepFlag = this.anticlockwise ? '0' : '1';
      var largeArcFlag;
      if ( this.angleDifference < Math.PI * 2 - epsilon ) {
        largeArcFlag = this.angleDifference < Math.PI ? '0' : '1';
        return 'A ' + this.radius + ' ' + this.radius + ' 0 ' + largeArcFlag + ' ' + sweepFlag + ' ' + this.end.x + ' ' + this.end.y;
      } else {
        // circle (or almost-circle) case needs to be handled differently
        // since SVG will not be able to draw (or know how to draw) the correct circle if we just have a start and end, we need to split it into two circular arcs
        
        // get the angle that is between and opposite of both of the points
        var splitOppositeAngle = ( this.startAngle + this.endAngle ) / 2; // this _should_ work for the modular case?
        var splitPoint = this.center.plus( Vector2.createPolar( this.radius, splitOppositeAngle ) );
        
        largeArcFlag = '0'; // since we split it in 2, it's always the small arc
        
        var firstArc = 'A ' + this.radius + ' ' + this.radius + ' 0 ' + largeArcFlag + ' ' + sweepFlag + ' ' + splitPoint.x + ' ' + splitPoint.y;
        var secondArc = 'A ' + this.radius + ' ' + this.radius + ' 0 ' + largeArcFlag + ' ' + sweepFlag + ' ' + this.end.x + ' ' + this.end.y;
        
        return firstArc + ' ' + secondArc;
      }
    },
    
    strokeLeft: function( lineWidth ) {
      return [ new Piece.Arc( this.center, this.radius + this.anticlockwise ? -lineWidth / 2 : lineWidth / 2, this.startAngle, this.endAngle, this.anticlockwise ) ];
    },
    
    strokeRight: function( lineWidth ) {
      return [ new Piece.Arc( this.center, this.radius + this.anticlockwise ? lineWidth / 2 : -lineWidth / 2, this.startAngle, this.endAngle, this.anticlockwise ) ];
    },
    
    intersectsBounds: function( bounds ) {
      throw new Error( 'Segment.intersectsBounds unimplemented!' );
    },
    
    // returns the resultant winding number of this ray intersecting this segment.
    windingIntersection: function( ray ) {
      // left here, if in the future we want to better-handle boundary points
      var epsilon = 0;
      
      // Run a general circle-intersection routine, then we can test the angles later.
      // Solves for the two solutions t such that ray.pos + ray.dir * t is on the circle.
      // Then we check whether the angle at each possible hit point is in our arc.
      var centerToRay = ray.pos.minus( this.center );
      var tmp = ray.dir.dot( centerToRay );
      var centerToRayDistSq = centerToRay.magnitudeSquared();
      var discriminant = 4 * tmp * tmp - 4 * ( centerToRayDistSq - this.radius * this.radius );
      if ( discriminant < epsilon ) {
        // ray misses circle entirely
        return 0;
      }
      var base = ray.dir.dot( this.center ) - ray.dir.dot( ray.pos );
      var sqt = Math.sqrt( discriminant ) / 2;
      var ta = base - sqt;
      var tb = base + sqt;
      
      if ( tb < epsilon ) {
        // circle is behind ray
        return 0;
      }
      
      var pointB = ray.pointAtDistance( tb );
      var normalB = pointB.minus( this.center ).normalized();
      
      var wind = 0;
      
      if ( ta < epsilon ) {
        // we are inside the circle, so only one intersection is possible
        if ( this.containsAngle( normalB.angle() ) ) {
          wind += this.anticlockwise ? 1 : -1; // since we are inside, wind this way
        }
      }
      else {
        // two possible hits (outside circle)
        var pointA = ray.pointAtDistance( ta );
        var normalA = pointA.minus( this.center ).normalized();
        
        if ( this.containsAngle( normalA.angle() ) ) {
          wind += this.anticlockwise ? -1 : 1; // hit from outside
        }
        if ( this.containsAngle( normalB.angle() ) ) {
          wind += this.anticlockwise ? 1 : -1; // this is the far hit, which winds the opposite way
        }
      }
      
      return wind;
    }
  };
  
  return Segment.Arc;
} );
