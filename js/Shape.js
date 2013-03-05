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
        lineStyles = new LineStyles();
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
              throw new Error( 'stroked round lineJoin not implemented .. add arc/arcTo' );
              // break;
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
            throw new Error( 'stroked round lineCap not implemented .. add arc/arcTo' );
            // break;
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
        var closingSegment = alreadyClosed ? null : new Segment.Line( segments[segments.length-1].end, segments[0].start );
        
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
      this._strokedShapeStyles = new LineStyles( lineStyles ); // shallow copy, since we consider linestyles to be mutable
      
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
  
  // supports both arc( centerX, centerY, radius, startAngle, endAngle, anticlockwise ) and arc( center, radius, startAngle, endAngle, anticlockwise )
  Shape.arc = function( centerX, centerY, radius, startAngle, endAngle, anticlockwise ) {
    return new Shape().arc( centerX, centerY, radius, startAngle, endAngle, anticlockwise );
  };
  
  /*---------------------------------------------------------------------------*
  * Line styles used in rendering the shape. Affects the stroked shape
  *----------------------------------------------------------------------------*/
  
  Shape.LineStyles = function( args ) {
    if ( args === undefined ) {
      args = {};
    }
    this.lineWidth = args.lineWidth !== undefined ? args.lineWidth : 1;
    this.lineCap = args.lineCap !== undefined ? args.lineCap : 'butt'; // butt, round, square
    this.lineJoin = args.lineJoin !== undefined ? args.lineJoin : 'miter'; // miter, round, bevel
    this.miterLimit = args.miterLimit !== undefined ? args.miterLimit : 10; // see https://svgwg.org/svg2-draft/painting.html for miterLimit computations
  };
  var LineStyles = Shape.LineStyles;
  LineStyles.prototype = {
    constructor: LineStyles,
    
    equals: function( other ) {
      return this.lineWidth === other.lineWidth && this.lineCap === other.lineCap && this.lineJoin === other.lineJoin && this.miterLimit === other.miterLimit;
    }
  };
  
  // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#concept-path for the path / subpath canvas concept
  Shape.Subpath = function() {
    this.points = [];
    this.segments = [];
    this.closed = false;
  };
  var Subpath = Shape.Subpath;
  Subpath.prototype = {
    addPoint: function( point ) {
      this.points.push( point );
    },
    
    addSegment: function( segment ) {
      if ( segment !== InvalidZeroLengthSegment ) {
        this.segments.push( segment );
      }
    },
    
    close: function() {
      this.closed = true;
    },
    
    getLength: function() {
      return this.points.length;
    },
    
    getFirstPoint: function() {
      return _.first( this.points );
    },
    
    getLastPoint: function() {
      return _.last( this.points );
    },
    
    getFirstSegment: function() {
      return _.first( this.segments );
    },
    
    getLastSegment: function() {
      return _.last( this.segments );
    },
    
    isDrawable: function() {
      return this.segments.length > 0;
    },
    
    isClosed: function() {
      return this.closed;
    },
    
    hasClosingSegment: function() {
      return !this.getFirstPoint().equals( this.getLastPoint() );
    },
    
    getClosingSegment: function() {
      assert && assert( this.hasClosingSegment(), 'Implicit closing segment unnecessary on a fully closed path' );
      return new Segment.Line( this.getLastPoint(), this.getFirstPoint() );
    }
  };
  
  /*---------------------------------------------------------------------------*
  * Pieces
  *----------------------------------------------------------------------------*/
  
  Shape.Piece = {};
  var Piece = Shape.Piece;
  
  Piece.Close = function() {};
  Piece.Close.prototype = {
    constructor: Piece.Close,
    
    writeToContext: function( context ) {
      context.closePath();
    },
    
    transformed: function( matrix ) {
      return [this];
    },
    
    applyPiece: function( shape ) {
      if ( shape.hasSubpaths() ) {
        var previousPath = shape.getLastSubpath();
        var nextPath = new Subpath();
        
        previousPath.close();
        shape.addSubpath( nextPath );
        nextPath.addPoint( previousPath.getFirstPoint() );
      }
    }
  };
  
  Piece.MoveTo = function( point ) {
    this.point = point;
  };
  Piece.MoveTo.prototype = {
    constructor: Piece.MoveTo,
    
    writeToContext: function( context ) {
      context.moveTo( this.point.x, this.point.y );
    },
    
    transformed: function( matrix ) {
      return [new Piece.MoveTo( matrix.timesVector2( this.point ) )];
    },
    
    applyPiece: function( shape ) {
      var subpath = new Subpath();
      subpath.addPoint( this.point );
      shape.addSubpath( subpath );
    }
  };
  
  Piece.LineTo = function( point ) {
    this.point = point;
  };
  Piece.LineTo.prototype = {
    constructor: Piece.LineTo,
    
    writeToContext: function( context ) {
      context.lineTo( this.point.x, this.point.y );
    },
    
    transformed: function( matrix ) {
      return [new Piece.LineTo( matrix.timesVector2( this.point ) )];
    },
    
    applyPiece: function( shape ) {
      // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-lineto
      if ( shape.hasSubpaths() ) {
        var start = shape.getLastSubpath().getLastPoint();
        var end = this.point;
        var line = new Segment.Line( start, end );
        shape.getLastSubpath().addSegment( line );
        shape.getLastSubpath().addPoint( end );
        shape.bounds = shape.bounds.withPoint( start ).withPoint( end );
        assert && assert( !isNaN( shape.bounds.x() ) );
      } else {
        shape.ensure( this.point );
      }
    }
  };
  
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
      var quadratic = new Segment.Quadratic( start, this.controlPoint, this.point );
      shape.getLastSubpath().addSegment( quadratic );
      shape.getLastSubpath().addPoint( this.point );
      if ( !quadratic.invalid ) {
        shape.bounds = shape.bounds.union( quadratic.bounds );
      }
    }
  };
  
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
      
      var radius;
      if ( matrix.scaling().x !== matrix.scaling().y ) {
        throw new Error( 'unimplemented asymmetric scaling of an arc - please implement ellipse' );
      } else {
        radius = matrix.scaling().x * this.radius;
      }
      return [new Piece.Arc( matrix.timesVector2( this.center ), radius, startAngle, endAngle, anticlockwise )];
    },
    
    applyPiece: function( shape ) {
      // see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-arc
      
      // we are assuming that the normal conditions were already met (or exceptioned out) so that these actually work with canvas
      var startPoint = this.center.plus( p( this.radius * Math.cos( this.startAngle ), this.radius * Math.sin( this.startAngle ) ) );
      var endPoint =   this.center.plus( p( this.radius * Math.cos( this.endAngle ),   this.radius * Math.sin( this.endAngle ) ) );
      
      // if there is already a point on the subpath, and it is different than our starting point, draw a line between them
      if ( shape.hasSubpaths() && shape.getLastSubpath().getLength() > 0 && !startPoint.equals( shape.getLastSubpath().getLastPoint(), 0 ) ) {
        shape.getLastSubpath().addSegment( new Segment.Line( shape.getLastSubpath().getLastPoint(), startPoint ) );
      }
      
      if ( !shape.hasSubpaths() ) {
        shape.addSubpath( new Subpath() );
      }
      
      var arc = new Segment.Arc( this.center, this.radius, this.startAngle, this.endAngle, this.anticlockwise );
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
  
  Piece.Rect = function( upperLeft, lowerRight ) {
    this.upperLeft = upperLeft;
    this.lowerRight = lowerRight;
    this.x = this.upperLeft.x;
    this.y = this.upperLeft.y;
    this.width = this.lowerRight.x - this.x;
    this.height = this.lowerRight.y - this.y;
  };
  Piece.Rect.prototype = {
    constructor: Piece.Rect,
    
    writeToContext: function( context ) {
      context.rect( this.x, this.y, this.width, this.height );
    },
    
    transformed: function( matrix ) {
      var a = matrix.timesVector2( p( this.x, this.y ) );
      var b = matrix.timesVector2( p( this.x + this.width, this.y ) );
      var c = matrix.timesVector2( p( this.x + this.width, this.y + this.height ) );
      var d = matrix.timesVector2( p( this.x, this.y + this.height ) );
      return [new Piece.MoveTo( a ), new Piece.LineTo( b ), new Piece.LineTo( c ), new Piece.LineTo( d ), new Piece.Close(), new Piece.MoveTo( a )];
    },
    
    applyPiece: function( shape ) {
      var subpath = new Subpath();
      shape.addSubpath( subpath );
      subpath.addPoint( p( this.x, this.y ) );
      subpath.addPoint( p( this.x + this.width, this.y ) );
      subpath.addPoint( p( this.x + this.width, this.y + this.height ) );
      subpath.addPoint( p( this.x, this.y + this.height ) );
      subpath.addSegment( new Segment.Line( subpath.points[0], subpath.points[1] ) );
      subpath.addSegment( new Segment.Line( subpath.points[1], subpath.points[2] ) );
      subpath.addSegment( new Segment.Line( subpath.points[2], subpath.points[3] ) );
      subpath.close();
      shape.addSubpath( new Subpath() );
      shape.getLastSubpath().addPoint( p( this.x, this.y ) );
      shape.bounds = shape.bounds.withPoint( this.upperLeft ).withPoint( this.lowerRight );
      assert && assert( !isNaN( shape.bounds.x() ) );
    }
  };
  
  /*---------------------------------------------------------------------------*
  * Segments
  *----------------------------------------------------------------------------*/
  
  Shape.Segment = {
    /*
     * Will contain:
     * properties:
     * start        - start point of this segment
     * end          - end point of this segment
     * startTangent - the tangent vector (normalized) to the segment at the start, pointing in the direction of motion (from start to end)
     * endTangent   - the tangent vector (normalized) to the segment at the end, pointing in the direction of motion (from start to end)
     * bounds       - the bounding box for the segment
     *
     * methods:
     * toPieces            - returns an array of pieces that are equivalent to this segment, assuming start points are preserved
     *                          TODO: is toPieces that valuable? it doesn't seem to have a strict guarantee on checking what the last segment did right now
     * getSVGPathFragment  - returns a string containing the SVG path. assumes that the start point is already provided, so anything that calls this needs to put the M calls first
     * strokeLeft          - returns an array of pieces that will draw an offset curve on the logical left side
     * strokeRight         - returns an array of pieces that will draw an offset curve on the logical right side
     * intersectsBounds    - whether this segment intersects the specified bounding box (not just the segment's bounding box, but the actual segment)
     * windingIntersection - returns the winding number for intersection with a ray
     */
  };
  var Segment = Shape.Segment;
  // TODO: consider actually using a segment prototype if we need it sometime
  
  // for now, simple way of handling error cases internally
  var InvalidZeroLengthSegment = {
    invalid: true
  };
  
  Segment.Line = function( start, end ) {
    if ( start.equals( end, 0 ) ) {
      return InvalidZeroLengthSegment;
    }
    this.start = start;
    this.end = end;
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
  
  Segment.Quadratic = function( start, control, end, skipComputations ) {
    if ( start.equals( end, 0 ) && start.equals( control, 0 ) ) {
      return InvalidZeroLengthSegment;
    }
    this.start = start;
    this.control = control;
    this.end = end;
    
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
    derivativeAt: function( t ) {
      return this.control.minus( this.start ).times( 2 * ( 1 - t ) ).plus( this.end.minus( this.control ).times( 2 * t ) );
    },
    
    offsetTo: function( r, includeMove, reverse ) {
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
      
      return includeMove ? ( [ new Piece.MoveTo( offsetCurves[0].start ) ].concat( result ) ) : result;
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
      return this.offsetTo( -lineWidth / 2, false, false );
    },
    
    strokeRight: function( lineWidth ) {
      return this.offsetTo( lineWidth / 2, false, true );
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
        result += ray.dir.perpendicular().dot( this.derivativeAt( ta ) ) < 0 ? 1 : -1;
      }
      
      if ( bValid ) {
        result += ray.dir.perpendicular().dot( this.derivativeAt( tb ) ) < 0 ? 1 : -1;
      }
      
      return result;
    }
  };
  
  Segment.Cubic = function( start, control1, control2, end ) {
    if ( start.equals( end, 0 ) && start.equals( control1, 0 ) && start.equals( control2, 0 ) ) {
      return InvalidZeroLengthSegment;
    }
    this.start = start;
    this.control1 = control1;
    this.control2 = control2;
    this.end = end;
  };
  Segment.Cubic.prototype = {
    // position: (1 - t)^3*p0 + 3*(1 - t)^2*t*p1 + 3*(1 - t) t^2*p2 + t^3*p3
    // derivative: -3 p0 (1 - t)^2 + 3 p1 (1 - t)^2 - 6 p1 (1 - t) t + 6 p2 (1 - t) t - 3 p2 t^2 + 3 p3 t^2
  };
  
  Segment.Arc = function( center, radius, startAngle, endAngle, anticlockwise ) {
    if ( radius <= 0 || startAngle === endAngle ) {
      return InvalidZeroLengthSegment;
    }
    // constraints
    assert && assert( !( ( !anticlockwise && endAngle - startAngle <= -Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle <= -Math.PI * 2 ) ), 'Not handling arcs with start/end angles that show differences in-between browser handling' );
    assert && assert( !( ( !anticlockwise && endAngle - startAngle > Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle > Math.PI * 2 ) ), 'Not handling arcs with start/end angles that show differences in-between browser handling' );
    
    var isFullPerimeter = ( !anticlockwise && endAngle - startAngle >= Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle >= Math.PI * 2 );
    
    this.center = center;
    this.radius = radius;
    this.startAngle = startAngle;
    this.endAngle = endAngle;
    this.anticlockwise = anticlockwise;
    
    this.start = center.plus( Vector2.createPolar( radius, startAngle ) );
    this.end = center.plus( Vector2.createPolar( radius, endAngle ) );
    // TODO: double-check the clockwiseness, since we have to reverse it from the mathematically-correct version
    this.startTangent = Vector2.createPolar( 1, startAngle + anticlockwise ? Math.PI / 2 : -Math.PI / 2 );
    this.endTangent = Vector2.createPolar( 1, endAngle + anticlockwise ? Math.PI / 2 : -Math.PI / 2 );
    
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
      var det = 4 * tmp * tmp - 4 * ( centerToRayDistSq - this.radius * this.radius );
      if ( det < epsilon ) {
        // ray misses circle entirely
        return 0;
      }
      var base = ray.dir.dot( this.center ) - ray.dir.dot( ray.pos );
      var sqt = Math.sqrt( det ) / 2;
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
  
  // TODO: notes at http://www.w3.org/TR/SVG/implnote.html#PathElementImplementationNotes
  // Canvas notes at http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-ellipse
  Segment.EllipticalArc = function( center, radiusX, radiusY, rotation, startAngle, endAngle, anticlockwise ) {
    if ( radiusX === 0 || radiusY === 0 || startAngle === endAngle ) {
      return InvalidZeroLengthSegment;
    }
    
    if ( radiusX < radiusY ) {
      throw new Error( 'Not verified to work if radiusX < radiusY' );
    }
    
    // constraints shared with Segment.Arc
    assert && assert( !( ( !anticlockwise && endAngle - startAngle <= -Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle <= -Math.PI * 2 ) ), 'Not handling elliptical arcs with start/end angles that show differences in-between browser handling' );
    assert && assert( !( ( !anticlockwise && endAngle - startAngle > Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle > Math.PI * 2 ) ), 'Not handling elliptical arcs with start/end angles that show differences in-between browser handling' );
    
    var isFullPerimeter = ( !anticlockwise && endAngle - startAngle >= Math.PI * 2 ) || ( anticlockwise && startAngle - endAngle >= Math.PI * 2 );
    
    this.center = center;
    this.radiusX = radiusX;
    this.radiusY = radiusY;
    this.rotation = rotation;
    this.startAngle = startAngle;
    this.endAngle = endAngle;
    this.anticlockwise = anticlockwise;
    
    // TODO: start/end
    // TODO: tangents
    // TODO: bounds
    throw new Error( 'Segment.EllipticalArc constructor unimplemented' );
  };
  Segment.EllipticalArc.prototype = {
    constructor: Segment.EllipticalArc,
    
    containsAngle: function( angle ) {
      throw new Error( 'Segment.EllipticalArc.containsAngle unimplemented' );
    },
    
    toPieces: function() {
      throw new Error( 'Segment.EllipticalArc.toPieces unimplemented' );
    },
    
    getSVGPathFragment: function() {
      // see http://www.w3.org/TR/SVG/paths.html#PathDataEllipticalArcCommands for more info
      // rx ry x-axis-rotation large-arc-flag sweep-flag x y
      throw new Error( 'Segment.EllipticalArc.getSVGPathFragment unimplemented' );
    },
    
    strokeLeft: function( lineWidth ) {
      throw new Error( 'Segment.EllipticalArc.strokeLeft unimplemented' );
    },
    
    strokeRight: function( lineWidth ) {
      throw new Error( 'Segment.EllipticalArc.strokeRight unimplemented' );
    },
    
    intersectsBounds: function( bounds ) {
      throw new Error( 'Segment.EllipticalArc.intersectsBounds unimplemented' );
    },
    
    // returns the resultant winding number of this ray intersecting this segment.
    windingIntersection: function( ray ) {
      throw new Error( 'Segment.EllipticalArc.windingIntersection unimplemented' );
    }
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
  
  // intersection between the line from p1-p2 and the line from p3-p4
  function lineLineIntersection( p1, p2, p3, p4 ) {
    return p(
      ( ( p1.x * p2.y - p1.y * p2.x ) * ( p3.x - p4.x ) - ( p1.x - p2.x ) * ( p3.x * p4.y - p3.y * p4.x ) ) / ( ( p1.x - p2.x ) * ( p3.y - p4.y ) - ( p1.y - p2.y ) * ( p3.x - p4.x ) ),
      ( ( p1.x * p2.y - p1.y * p2.x ) * ( p3.y - p4.y ) - ( p1.y - p2.y ) * ( p3.x * p4.y - p3.y * p4.x ) ) / ( ( p1.x - p2.x ) * ( p3.y - p4.y ) - ( p1.y - p2.y ) * ( p3.x - p4.x ) )
    );
  }
  
  return Shape;
} );
