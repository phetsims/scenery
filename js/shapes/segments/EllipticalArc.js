// Copyright 2002-2012, University of Colorado

/**
 * Elliptical arc segment
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );

  var scenery = require( 'SCENERY/scenery' );
  
  var Vector2 = require( 'DOT/Vector2' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Transform3 = require( 'DOT/Transform3' );
  var toDegrees = require( 'DOT/Util' ).toDegrees;

  var Segment = require( 'SCENERY/shapes/segments/Segment' );
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );
  require( 'SCENERY/shapes/util/Subpath' );

  // TODO: notes at http://www.w3.org/TR/SVG/implnote.html#PathElementImplementationNotes
  // Canvas notes at http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-ellipse
  Segment.EllipticalArc = function( center, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) {
    this.center = center;
    this.radiusX = radiusX;
    this.radiusY = radiusY;
    this.rotation = rotation;
    this.startAngle = startAngle;
    this.endAngle = endAngle;
    this.anticlockwise = anticlockwise;
    
    this.unitTransform = Segment.EllipticalArc.computeUnitTransform( center, radiusX, radiusY, rotation );
    
    this.start = this.positionAtAngle( startAngle );
    this.end = this.positionAtAngle( endAngle );
    this.startTangent = this.tangentAtAngle( startAngle );
    this.endTangent = this.tangentAtAngle( endAngle );
    
    if ( radiusX === 0 || radiusY === 0 || startAngle === endAngle ) {
      this.invalid = true;
      return;
    }
    
    if ( radiusX < radiusY ) {
      // TODO: check this
      throw new Error( 'Not verified to work if radiusX < radiusY' );
    }
    
    // constraints shared with Segment.Arc
    assert && assert( !( ( !anticlockwise && endAngle - startAngle <= -Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle <= -Math.PI * 2 ) ), 'Not handling elliptical arcs with start/end angles that show differences in-between browser handling' );
    assert && assert( !( ( !anticlockwise && endAngle - startAngle > Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle > Math.PI * 2 ) ), 'Not handling elliptical arcs with start/end angles that show differences in-between browser handling' );
    
    var isFullPerimeter = ( !anticlockwise && endAngle - startAngle >= Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle >= Math.PI * 2 );
    
    // compute an angle difference that represents how "much" of the circle our arc covers
    this.angleDifference = this.anticlockwise ? this.startAngle - this.endAngle : this.endAngle - this.startAngle;
    if ( this.angleDifference < 0 ) {
      this.angleDifference += Math.PI * 2;
    }
    assert && assert( this.angleDifference >= 0 ); // now it should always be zero or positive
    
    // a unit arg segment that we can map to our ellipse. useful for hit testing and such.
    this.unitArcSegment = new Segment.Arc( center, 1, startAngle, endAngle, anticlockwise );
    
    this.bounds = Bounds2.NOTHING;
    this.bounds = this.bounds.withPoint( this.start );
    this.bounds = this.bounds.withPoint( this.end );
    
    // for bounds computations
    var that = this;
    function boundsAtAngle( angle ) {
      if ( that.containsAngle( angle ) ) {
        // the boundary point is in the arc
        that.bounds = that.bounds.withPoint( that.positionAtAngle( angle ) );
      }
    }
    
    // if the angles are different, check extrema points
    if ( startAngle !== endAngle ) {
      // solve the mapping from the unit circle, find locations where a coordinate of the gradient is zero.
      // we find one extrema point for both x and y, since the other two are just rotated by pi from them.
      var xAngle = Math.atan( -( radiusY / radiusX ) * Math.tan( rotation ) );
      var yAngle = Math.atan( ( radiusY / radiusX ) / Math.tan( rotation ) );
      
      // check all of the extrema points
      boundsAtAngle( xAngle );
      boundsAtAngle( xAngle + Math.PI );
      boundsAtAngle( yAngle );
      boundsAtAngle( yAngle + Math.PI );
    }
  };
  Segment.EllipticalArc.prototype = {
    constructor: Segment.EllipticalArc,
    
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
      return this.unitTransform.transformPosition2( Vector2.createPolar( 1, angle ) );
    },
    
    tangentAtAngle: function( angle ) {
      return this.unitTransform.transformDelta2( Vector2.createPolar( 1, angle + this.anticlockwise ? Math.PI / 2 : -Math.PI / 2 ) );
    },
    
    // TODO: refactor? exact same as Segment.Arc
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
      return [ new Piece.EllipticalArc( this.center, this.radiusX, this.radiusY, this.rotation, this.startAngle, this.endAngle, this.anticlockwise ) ];
    },
    
    // discretizes the elliptical arc and returns an offset curve as a list of lineTos
    offsetTo: function( r, reverse ) {
      // how many segments to create (possibly make this more adaptive?)
      var quantity = 32;
      
      var result = [];
      for ( var i = 1; i < quantity; i++ ) {
        var ratio = i - ( quantity - 1 );
        if ( reverse ) {
          ratio = 1 - ratio;
        }
        var angle = this.startAngle + ratio * ( this.endAngle - this.startAngle );
        
        var point = this.positionAtAngle( angle ).plus( this.tangentAtAngle( angle ).perpendicular().normalized().times( r ) );
        result.push( new Piece.LineTo( point ) );
      }
      
      return result;
    },
    
    getSVGPathFragment: function() {
      // see http://www.w3.org/TR/SVG/paths.html#PathDataEllipticalArcCommands for more info
      // rx ry x-axis-rotation large-arc-flag sweep-flag x y
      var epsilon = 0.01; // allow some leeway to render things as 'almost circles'
      var sweepFlag = this.anticlockwise ? '0' : '1';
      var largeArcFlag;
      var degreesRotation = toDegrees( this.rotation ); // bleh, degrees?
      if ( this.angleDifference < Math.PI * 2 - epsilon ) {
        largeArcFlag = this.angleDifference < Math.PI ? '0' : '1';
        return 'A ' + this.radiusX + ' ' + this.radiusY + ' ' + degreesRotation + ' ' + largeArcFlag + ' ' + sweepFlag + ' ' + this.end.x + ' ' + this.end.y;
      } else {
        // ellipse (or almost-ellipse) case needs to be handled differently
        // since SVG will not be able to draw (or know how to draw) the correct circle if we just have a start and end, we need to split it into two circular arcs
        
        // get the angle that is between and opposite of both of the points
        var splitOppositeAngle = ( this.startAngle + this.endAngle ) / 2; // this _should_ work for the modular case?
        var splitPoint = this.positionAtAngle( splitOppositeAngle );
        
        largeArcFlag = '0'; // since we split it in 2, it's always the small arc
        
        var firstArc = 'A ' + this.radiusX + ' ' + this.radiusY + ' ' + degreesRotation + ' ' + largeArcFlag + ' ' + sweepFlag + ' ' + splitPoint.x + ' ' + splitPoint.y;
        var secondArc = 'A ' + this.radiusX + ' ' + this.radiusY + ' ' + degreesRotation + ' ' + largeArcFlag + ' ' + sweepFlag + ' ' + this.end.x + ' ' + this.end.y;
        
        return firstArc + ' ' + secondArc;
      }
    },
    
    strokeLeft: function( lineWidth ) {
      return this.offsetTo( -lineWidth / 2, false );
    },
    
    strokeRight: function( lineWidth ) {
      return this.offsetTo( lineWidth / 2, true );
    },
    
    intersectsBounds: function( bounds ) {
      throw new Error( 'Segment.EllipticalArc.intersectsBounds unimplemented' );
    },
    
    // returns the resultant winding number of this ray intersecting this segment.
    windingIntersection: function( ray ) {
      // be lazy. transform it into the space of a non-elliptical arc.
      var rayInUnitCircleSpace = this.unitTransform.inverseRay2( ray );
      return this.unitArcSegment.windingIntersection( rayInUnitCircleSpace );
    }
  };
  
  // adapted from http://www.w3.org/TR/SVG/implnote.html#PathElementImplementationNotes
  // transforms the unit circle onto our ellipse
  Segment.EllipticalArc.computeUnitTransform = function( center, radiusX, radiusY, rotation ) {
    return new Transform3( Matrix3.translation( center.x, center.y ) // TODO: convert to Matrix3.translation( this.center) when available
                                  .timesMatrix( Matrix3.rotation2( rotation ) )
                                  .timesMatrix( Matrix3.scaling( radiusX, radiusY ) ) );
  };
  
  return Segment.EllipticalArc;
} );
