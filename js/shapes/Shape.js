// Copyright 2002-2012, University of Colorado

/**
 * Shape handling
 *
 * Shapes are internally made up of pieces (generally individual Canvas calls),
 * which for simplicity of stroking and hit testing are then broken up into
 * individual segments stored in subpaths. Familiarity with how Canvas handles
 * subpaths is helpful for understanding this code.
 *
 * Canvas spec: http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html
 * SVG spec: http://www.w3.org/TR/SVG/expanded-toc.html
 *           http://www.w3.org/TR/SVG/paths.html#PathData (for paths)
 * Notes for elliptical arcs: http://www.w3.org/TR/SVG/implnote.html#PathElementImplementationNotes
 * Notes for painting strokes: https://svgwg.org/svg2-draft/painting.html
 *
 * TODO: add nonzero / evenodd support when browsers support it
 * TODO: docs
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  var assertExtra = require( 'ASSERT/assert' )( 'scenery.extra', true );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Vector2 = require( 'DOT/Vector2' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Ray2 = require( 'DOT/Ray2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Transform3 = require( 'DOT/Transform3' );
  var toDegrees = require( 'DOT/Util' ).toDegrees;
  var lineLineIntersection = require( 'DOT/Util' ).lineLineIntersection;
  
  var Subpath = require( 'SCENERY/shapes/util/Subpath' );
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );
  require( 'SCENERY/shapes/util/LineStyles' );
  require( 'SCENERY/shapes/pieces/Arc' );
  require( 'SCENERY/shapes/pieces/Close' );
  require( 'SCENERY/shapes/pieces/CubicCurveTo' );
  require( 'SCENERY/shapes/pieces/EllipticalArc' );
  require( 'SCENERY/shapes/pieces/LineTo' );
  require( 'SCENERY/shapes/pieces/MoveTo' );
  require( 'SCENERY/shapes/pieces/QuadraticCurveTo' );
  require( 'SCENERY/shapes/pieces/Rect' );
  require( 'SCENERY/shapes/segments/Line' );
  
  // for brevity
  function p( x,y ) { return new Vector2( x, y ); }
  
  // a normalized vector for non-zero winding checks
  // var weirdDir = p( Math.PI, 22 / 7 );
  
  scenery.Shape = function( pieces, optionalClose ) {
    // higher-level Canvas-esque drawing commands
    this.pieces = [];
    
    // lower-level piecewise mathematical description using segments
    this.subpaths = [];
    
    // computed bounds for all pieces added so far
    this.bounds = Bounds2.NOTHING;
    
    // cached stroked shape (so hit testing can be done quickly on stroked shapes)
    this._strokedShape = null;
    this._strokedShapeComputed = false;
    this._strokedShapeStyles = null;
    
    var that = this;
    // initialize with pieces passed in
    if ( pieces !== undefined ) {
      _.each( pieces, function( piece ) {
        that.addPiece( piece );
      } );
    }
    if ( optionalClose ) {
      this.addPiece( new Piece.Close() );
    }
  };
  var Shape = scenery.Shape;
  
  Shape.prototype = {
    constructor: Shape,
    
    moveTo: function( x, y ) {
      // moveTo( point )
      if ( y === undefined && typeof x === 'object' ) {
        // wrap it in a Vector2 if the class doesn't match
        var point = x instanceof Vector2 ? x : new Vector2( x.x, x.y );
        this.addPiece( new Piece.MoveTo( point ) );
      } else { // moveTo( x, y )
        this.addPiece( new Piece.MoveTo( p( x, y ) ) );
      }
      return this;
    },
    
    lineTo: function( x, y ) {
      // lineTo( point )
      if ( y === undefined && typeof x === 'object' ) {
        // wrap it in a Vector2 if the class doesn't match
        var point = x instanceof Vector2 ? x : new Vector2( x.x, x.y );
        this.addPiece( new Piece.LineTo( point ) );
      } else { // lineTo( x, y )
        this.addPiece( new Piece.LineTo( p( x, y ) ) );
      }
      return this;
    },
    
    quadraticCurveTo: function( cpx, cpy, x, y ) {
      // quadraticCurveTo( control, point )
      if ( x === undefined && typeof cpx === 'object' ) {
        // wrap it in a Vector2 if the class doesn't match
        var controlPoint = cpx instanceof Vector2 ? cpx : new Vector2( cpx.x, cpx.y );
        var point = cpy instanceof Vector2 ? cpy : new Vector2( cpy.x, cpy.y );
        this.addPiece( new Piece.QuadraticCurveTo( controlPoint, point ) );
      } else { // quadraticCurveTo( cpx, cpy, x, y )
        this.addPiece( new Piece.QuadraticCurveTo( p( cpx, cpy ), p( x, y ) ) );
      }
      return this;
    },
    
    cubicCurveTo: function( cp1x, cp1y, cp2x, cp2y, x, y ) {
      // cubicCurveTo( cp1, cp2, end )
      if ( cp2y === undefined && typeof cp1x === 'object' ) {
        // wrap it in a Vector2 if the class doesn't match
        var control1 = cp1x instanceof Vector2 ? cp1x : new Vector2( cp1x.x, cp1x.y );
        var control2 = cp1y instanceof Vector2 ? cp1y : new Vector2( cp1y.x, cp1y.y );
        var end = cp2x instanceof Vector2 ? cp2x : new Vector2( cp2x.x, cp2x.y );
        this.addPiece( new Piece.CubicCurveTo( control1, control2, end ) );
      } else { // cubicCurveTo( cp1x, cp1y, cp2x, cp2y, x, y )
        this.addPiece( new Piece.CubicCurveTo( p( cp1x, cp1y ), p( cp2x, cp2y ), p( x, y ) ) );
      }
      return this;
    },
    
    /*
     * Draws a circle using the arc() call with the following parameters:
     * circle( center, radius ) // center is a Vector2
     * circle( centerX, centerY, radius )
     */
    circle: function( centerX, centerY, radius ) {
      if ( typeof centerX === 'object' ) {
        // circle( center, radius )
        var center = centerX;
        radius = centerY;
        return this.arc( center, radius, 0, Math.PI * 2, false );
      } else {
        // circle( centerX, centerY, radius )
        return this.arc( p( centerX, centerY ), radius, 0, Math.PI * 2, false );
      }
    },
    
    /*
     * Draws an ellipse using the ellipticalArc() call with the following parameters:
     * ellipse( center, radiusX, radiusY, rotation ) // center is a Vector2
     * ellipse( centerX, centerY, radiusX, radiusY, rotation )
     */
    ellipse: function( centerX, centerY, radiusX, radiusY, rotation ) {
      // TODO: Ellipse/EllipticalArc has a mess of parameters. Consider parameter object, or double-check parameter handling
      if ( typeof centerX === 'object' ) {
        // ellipse( center, radiusX, radiusY, rotation )
        var center = centerX;
        rotation = radiusY;
        radiusY = radiusX;
        radiusX = centerY;
        return this.ellipticalArc( center, radiusX, radiusY, rotation || 0, 0, Math.PI * 2, false );
      } else {
        // ellipse( centerX, centerY, radiusX, radiusY, rotation )
        return this.ellipticalArc( p( centerX, centerY ), radiusX, radiusY, rotation || 0, 0, Math.PI * 2, false );
      }
    },
    
    /*
     * Draws an arc using the Canvas 2D semantics, with the following parameters:
     * arc( center, radius, startAngle, endAngle, anticlockwise )
     * arc( centerX, centerY, radius, startAngle, endAngle, anticlockwise )
     */
    arc: function( centerX, centerY, radius, startAngle, endAngle, anticlockwise ) {
      if ( typeof centerX === 'object' ) {
        // arc( center, radius, startAngle, endAngle, anticlockwise )
        anticlockwise = endAngle;
        endAngle = startAngle;
        startAngle = radius;
        radius = centerY;
        var center = centerX;
        this.addPiece( new Piece.Arc( center, radius, startAngle, endAngle, anticlockwise ) );
      } else {
        // arc( centerX, centerY, radius, startAngle, endAngle, anticlockwise )
        this.addPiece( new Piece.Arc( p( centerX, centerY ), radius, startAngle, endAngle, anticlockwise ) );
      }
      return this;
    },
    
    /*
     * Draws an elliptical arc using the Canvas 2D semantics, with the following parameters:
     * ellipticalArc( center, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise )
     * ellipticalArc( centerX, centerY, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise )
     */
    ellipticalArc: function( centerX, centerY, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) {
      // TODO: Ellipse/EllipticalArc has a mess of parameters. Consider parameter object, or double-check parameter handling
      if ( typeof centerX === 'object' ) {
        // ellipticalArc( center, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise )
        anticlockwise = endAngle;
        endAngle = startAngle;
        startAngle = rotation;
        rotation = radiusY;
        radiusY = radiusX;
        radiusX = centerY;
        var center = centerX;
        this.addPiece( new Piece.EllipticalArc( center, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) );
      } else {
        // ellipticalArc( centerX, centerY, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise )
        this.addPiece( new Piece.EllipticalArc( p( centerX, centerY ), radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) );
      }
      return this;
    },
    
    rect: function( a, b, c, d ) {
      // rect( upperLeft, lowerRight )
      if ( c === undefined && typeof a === 'object' && typeof b === 'object' ) {
        // wrap it in a Vector2 if the class doesn't match
        var upperLeft = a instanceof Vector2 ? a : new Vector2( a.x, a.y );
        var lowerRight = b instanceof Vector2 ? b : new Vector2( b.x, b.y );
        this.addPiece( new Piece.Rect( upperLeft, lowerRight ) );
      } else {
        // rect( x, y, width, height )
        this.addPiece( new Piece.Rect( p( a, b ), p( a + c, b + d ) ) );
      }
      return this;
    },

    //Create a round rectangle. All arguments are number.
    //Rounding is currently using quadraticCurveTo.  Please note, future versions may use arcTo
    //TODO: rewrite with arcTo?
    roundRect: function( x, y, width, height, arcw, arch ) {
      this.moveTo( x + arcw, y ).
          lineTo( x + width - arcw, y ).
          quadraticCurveTo( x + width, y, x + width, y + arch ).
          lineTo( x + width, y + height - arch ).
          quadraticCurveTo( x + width, y + height, x + width - arcw, y + height ).
          lineTo( x + arcw, y + height ).
          quadraticCurveTo( x, y + height, x, y + height - arch ).
          lineTo( x, y + arch ).
          quadraticCurveTo( x, y, x + arcw, y );
      return this;
    },
    
    close: function() {
      this.addPiece( new Piece.Close() );
      return this;
    },
    
    addPiece: function( piece ) {
      this.pieces.push( piece );
      piece.applyPiece( this );
      this.invalidate();
    },
    
    // write out this shape's path to a canvas 2d context. does NOT include the beginPath()!
    writeToContext: function( context ) {
      _.each( this.pieces, function( piece ) {
        piece.writeToContext( context );
      } );
    },
    
    // returns something like "M150 0 L75 200 L225 200 Z" for a triangle
    getSVGPath: function() {
      var subpathStrings = [];
      _.each( this.subpaths, function( subpath ) {
        if( subpath.isDrawable() ) {
          // since the commands after this are relative to the previous 'point', we need to specify a move to the initial point
          var startPoint = subpath.getFirstSegment().start;
          assert && assert( startPoint.equals( subpath.getFirstPoint(), 0.00001 ) ); // sanity check
          var string = 'M ' + startPoint.x + ' ' + startPoint.y + ' ';
          
          string += _.map( subpath.segments, function( segment ) { return segment.getSVGPathFragment(); } ).join( ' ' );
          
          if ( subpath.isClosed() ) {
            string += ' Z';
          }
          subpathStrings.push( string );
        }
      } );
      return subpathStrings.join( ' ' );
    },
    
    // return a new Shape that is transformed by the associated matrix
    transformed: function( matrix ) {
      return new Shape( _.flatten( _.map( this.pieces, function( piece ) { return piece.transformed( matrix ); } ), true ) );
    },
    
    // returns the bounds. if lineStyles exists, include the stroke in the bounds
    // TODO: consider renaming to getBounds()?
    computeBounds: function( lineStyles ) {
      if ( lineStyles ) {
        return this.bounds.union( this.getStrokedShape( lineStyles ).bounds );
      } else {
        return this.bounds;
      }
    },
    
    containsPoint: function( point ) {
      var wind = 0;
      
      // we pick a ray, and determine the winding number over that ray. if the number of segments crossing it CCW == number of segments crossing it CW, then the point is contained in the shape
      var ray = new Ray2( point, p( 1, 0 ) );
      
      _.each( this.subpaths, function( subpath ) {
        if ( subpath.isDrawable() ) {
          _.each( subpath.segments, function( segment ) {
            wind += segment.windingIntersection( ray );
          } );
          
          // handle the implicit closing line segment
          if ( subpath.hasClosingSegment() ) {
            wind += subpath.getClosingSegment().windingIntersection( ray );
          }
        }
      } );
      return wind !== 0;
    },
    
    intersectsBounds: function( bounds ) {
      var intersects = false;
      // TODO: break-out-early optimizations
      _.each( this.subpaths, function( subpath ) {
        if ( subpath.isDrawable() ) {
          _.each( subpath.segments, function( segment ) {
            intersects = intersects && segment.intersectsBounds( bounds );
          } );
          
          // handle the implicit closing line segment
          if ( subpath.hasClosingSegment() ) {
            intersects = intersects && subpath.getClosingSegment().intersectsBounds( bounds );
          }
        }
      } );
      return intersects;
    },
    
    invalidate: function() {
      this._strokedShapeComputed = false;
    },
    
    // returns a new Shape that is an outline of the stroked path of this current Shape. currently not intended to be nested (doesn't do intersection computations yet)
    getStrokedShape: function( lineStyles ) {
      
      if ( lineStyles === undefined ) {
        lineStyles = new scenery.LineStyles();
      }
      
      // return a cached version if possible
      if ( this._strokedShapeComputed && this._strokedShapeStyles.equals( lineStyles ) ) {
        return this._strokedShape;
      }
      
      // filter out subpaths where nothing would be drawn
      var subpaths = _.filter( this.subpaths, function( subpath ) { return subpath.isDrawable(); } );
      
      var shape = new Shape();
      
      var lineWidth = lineStyles.lineWidth;
      
      // joins two segments together on the logical "left" side, at 'center' (where they meet), and normalized tangent vectors in the direction of the stroking
      // to join on the "right" side, switch the tangent order and negate them
      function join( center, fromTangent, toTangent ) {
        // where our join path starts and ends
        var fromPoint = center.plus( fromTangent.perpendicular().negated().times( lineWidth / 2 ) );
        var toPoint = center.plus( toTangent.perpendicular().negated().times( lineWidth / 2 ) );
        
        // only insert a join on the non-acute-angle side
        if ( fromTangent.perpendicular().dot( toTangent ) > 0 ) {
          switch( lineStyles.lineJoin ) {
            case 'round':
              var fromAngle = fromTangent.angle() + Math.PI / 2;
              var toAngle = toTangent.angle() + Math.PI / 2;
              shape.addPiece( new Piece.Arc( center, lineWidth / 2, fromAngle, toAngle, true ) );
              break;
            case 'miter':
              var theta = fromTangent.angleBetween( toTangent.negated() );
              if ( 1 / Math.sin( theta / 2 ) <= lineStyles.miterLimit ) {
                // draw the miter
                var miterPoint = lineLineIntersection( fromPoint, fromPoint.plus( fromTangent ), toPoint, toPoint.plus( toTangent ) );
                shape.addPiece( new Piece.LineTo( miterPoint ) );
                shape.addPiece( new Piece.LineTo( toPoint ) );
              } else {
                // angle too steep, use bevel instead. same as below, but copied for linter
                shape.addPiece( new Piece.LineTo( toPoint ) );
              }
              break;
            case 'bevel':
              shape.addPiece( new Piece.LineTo( toPoint ) );
              break;
          }
        } else {
          // no join necessary here since we have the acute angle. just simple lineTo for now so that the next segment starts from the right place
          // TODO: can we prevent self-intersection here?
          if ( !fromPoint.equals( toPoint ) ) {
            shape.addPiece( new Piece.LineTo( toPoint ) );
          }
        }
      }
      
      // draws the necessary line cap from the endpoint 'center' in the direction of the tangent
      function cap( center, tangent ) {
        switch( lineStyles.lineCap ) {
          case 'butt':
            shape.addPiece( new Piece.LineTo( center.plus( tangent.perpendicular().times( lineWidth / 2 ) ) ) );
            break;
          case 'round':
            var tangentAngle = tangent.angle();
            shape.addPiece( new Piece.Arc( center, lineWidth / 2, tangentAngle + Math.PI / 2, tangentAngle - Math.PI / 2, true ) );
            break;
          case 'square':
            var toLeft = tangent.perpendicular().negated().times( lineWidth / 2 );
            var toRight = tangent.perpendicular().times( lineWidth / 2 );
            var toFront = tangent.times( lineWidth / 2 );
            shape.addPiece( new Piece.LineTo( center.plus( toLeft ).plus( toFront ) ) );
            shape.addPiece( new Piece.LineTo( center.plus( toRight ).plus( toFront ) ) );
            shape.addPiece( new Piece.LineTo( center.plus( toRight ) ) );
            break;
        }
      }
      
      _.each( subpaths, function( subpath ) {
        var i;
        var segments = subpath.segments;
        
        // TODO: shortcuts for _.first( segments ) and _.last( segments ),
        
        // we don't need to insert an implicit closing segment if the start and end points are the same
        var alreadyClosed = _.last( segments ).end.equals( _.first( segments ).start );
        // if there is an implicit closing segment
        var closingSegment = alreadyClosed ? null : new scenery.Segment.Line( segments[segments.length-1].end, segments[0].start );
        
        // move to the first point in our stroked path
        shape.addPiece( new Piece.MoveTo( segmentStartLeft( _.first( segments ), lineWidth ) ) );
        
        // stroke the logical "left" side of our path
        for ( i = 0; i < segments.length; i++ ) {
          if ( i > 0 ) {
            join( segments[i].start, segments[i-1].endTangent, segments[i].startTangent, true );
          }
          _.each( segments[i].strokeLeft( lineWidth ), function( piece ) {
            shape.addPiece( piece );
          } );
        }
        
        // handle the "endpoint"
        if ( subpath.closed ) {
          if ( alreadyClosed ) {
            join( _.last( segments ).end, _.last( segments ).endTangent, _.first( segments ).startTangent );
            shape.addPiece( new Piece.Close() );
            shape.addPiece( new Piece.MoveTo( segmentStartRight( _.first( segments ), lineWidth ) ) );
            join( _.last( segments ).end, _.first( segments ).startTangent.negated(), _.last( segments ).endTangent.negated() );
          } else {
            // logical "left" stroke on the implicit closing segment
            join( closingSegment.start, _.last( segments ).endTangent, closingSegment.startTangent );
            _.each( closingSegment.strokeLeft( lineWidth ), function( piece ) {
              shape.addPiece( piece );
            } );
            
            // TODO: similar here to other block of if.
            join( closingSegment.end, closingSegment.endTangent, _.first( segments ).startTangent );
            shape.addPiece( new Piece.Close() );
            shape.addPiece( new Piece.MoveTo( segmentStartRight( _.first( segments ), lineWidth ) ) );
            join( closingSegment.end, _.first( segments ).startTangent.negated(), closingSegment.endTangent.negated() );
            
            // logical "right" stroke on the implicit closing segment
            _.each( closingSegment.strokeRight( lineWidth ), function( piece ) {
              shape.addPiece( piece );
            } );
            join( closingSegment.start, closingSegment.startTangent.negated(), _.last( segments ).endTangent.negated() );
          }
        } else {
          cap( _.last( segments ).end, _.last( segments ).endTangent );
        }
        
        // stroke the logical "right" side of our path
        for ( i = segments.length - 1; i >= 0; i-- ) {
          if ( i < segments.length - 1 ) {
            join( segments[i].end, segments[i+1].startTangent.negated(), segments[i].endTangent.negated(), false );
          }
          _.each( segments[i].strokeRight( lineWidth ), function( piece ) {
            shape.addPiece( piece );
          } );
        }
        
        // handle the start point
        if ( subpath.closed ) {
          // we already did the joins, just close the 'right' side
          shape.addPiece( new Piece.Close() );
        } else {
          cap( _.first( segments ).start, _.first( segments ).startTangent.negated() );
          shape.addPiece( new Piece.Close() );
        }
      } );
      
      this._strokedShape = shape;
      this._strokedShapeComputed = true;
      this._strokedShapeStyles = new scenery.LineStyles( lineStyles ); // shallow copy, since we consider linestyles to be mutable
      
      return shape;
    },
    
    /*---------------------------------------------------------------------------*
    * Internal subpath computations
    *----------------------------------------------------------------------------*/
    
    ensure: function( point ) {
      if ( !this.hasSubpaths() ) {
        this.addSubpath( new Subpath() );
        this.getLastSubpath().addPoint( point );
      }
    },
    
    addSubpath: function( subpath ) {
      this.subpaths.push( subpath );
    },
    
    hasSubpaths: function() {
      return this.subpaths.length > 0;
    },
    
    getLastSubpath: function() {
      return _.last( this.subpaths );
    }
  };
  
  /*---------------------------------------------------------------------------*
  * Shape shortcuts
  *----------------------------------------------------------------------------*/
  
  Shape.rectangle = function( x, y, width, height ) {
    return new Shape().rect( x, y, width, height );
  };
  Shape.rect = Shape.rectangle;

  //Create a round rectangle. All arguments are number.
  //Rounding is currently using quadraticCurveTo.  Please note, future versions may use arcTo
  //TODO: rewrite with arcTo?
  Shape.roundRect = function( x, y, width, height, arcw, arch ) {
    return new Shape().roundRect( x, y, width, height, arcw, arch );
  };
  Shape.roundRectangle = Shape.roundRect;
  
  Shape.bounds = function( bounds ) {
    return new Shape().rect( p( bounds.minX, bounds.minY ), p( bounds.maxX, bounds.maxY ) );
  };
  
  Shape.lineSegment = function( a, b ) {
    // TODO: add type assertions?
    return new Shape().moveTo( a ).lineTo( b );
  };
  
  Shape.regularPolygon = function( sides, radius ) {
    var first = true;
    return new Shape( _.map( _.range( sides ), function( k ) {
      var theta = 2 * Math.PI * k / sides;
      if ( first ) {
        first = false;
        // first segment should be a moveTo
        return new Piece.MoveTo( p( radius * Math.cos( theta ), radius * Math.sin( theta ) ) );
      } else {
        return new Piece.LineTo( p( radius * Math.cos( theta ), radius * Math.sin( theta ) ) );
      }
    } ), true );
  };
  
  // supports both circle( centerX, centerY, radius ), circle( center, radius ), and circle( radius ) with the center default to 0,0
  Shape.circle = function( centerX, centerY, radius ) {
    if ( centerY === undefined ) {
      // circle( radius ), center = 0,0
      return new Shape().circle( 0, 0, centerX );
    }
    return new Shape().circle( centerX, centerY, radius ).close();
  };
  
  /*
   * Supports ellipse( centerX, centerY, radiusX, radiusY ), ellipse( center, radiusX, radiusY ), and ellipse( radiusX, radiusY )
   * with the center default to 0,0 and rotation of 0
   */
  Shape.ellipse = function( centerX, centerY, radiusX, radiusY ) {
    // TODO: Ellipse/EllipticalArc has a mess of parameters. Consider parameter object, or double-check parameter handling
    if ( radiusX === undefined ) {
      // ellipse( radiusX, radiusY ), center = 0,0
      return new Shape().ellipse( 0, 0, centerX, centerY );
    }
    return new Shape().ellipse( centerX, centerY, radiusX, radiusY ).close();
  };
  
  // supports both arc( centerX, centerY, radius, startAngle, endAngle, anticlockwise ) and arc( center, radius, startAngle, endAngle, anticlockwise )
  Shape.arc = function( centerX, centerY, radius, startAngle, endAngle, anticlockwise ) {
    return new Shape().arc( centerX, centerY, radius, startAngle, endAngle, anticlockwise );
  };
  
  
  // TODO: performance / cleanliness to have these as methods instead?
  function segmentStartLeft( segment, lineWidth ) {
    assert && assert( lineWidth !== undefined );
    return segment.start.plus( segment.startTangent.perpendicular().negated().times( lineWidth / 2 ) );
  }
  
  function segmentEndLeft( segment, lineWidth ) {
    assert && assert( lineWidth !== undefined );
    return segment.end.plus( segment.endTangent.perpendicular().negated().times( lineWidth / 2 ) );
  }
  
  function segmentStartRight( segment, lineWidth ) {
    assert && assert( lineWidth !== undefined );
    return segment.start.plus( segment.startTangent.perpendicular().times( lineWidth / 2 ) );
  }
  
  function segmentEndRight( segment, lineWidth ) {
    assert && assert( lineWidth !== undefined );
    return segment.end.plus( segment.endTangent.perpendicular().times( lineWidth / 2 ) );
  }
  
  return Shape;
} );
