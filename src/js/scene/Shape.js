// Copyright 2002-2012, University of Colorado

/**
 * Shape handling
 *
 * TODO: add nonzero / evenodd support when browsers support it
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    
    var Vector2 = phet.math.Vector2;
    
    // for brevity
    function p( x,y ) { return new Vector2( x, y ); }
    
    phet.scene.Shape = function( pieces, optionalClose ) {
        this.pieces = [];
        
        this._strokedShape = null;
        this._strokedShapeComputed = false;
        this._strokedShapeStyles = null;
        
        this.subpaths = [];
        this.bounds = phet.math.Bounds2.NOTHING;
        
        var that = this;
        
        // initialize with pieces passed in
        if( pieces !== undefined ) {
            _.each( pieces, function( piece ) {
                that.addPiece( piece );
            } );
        }
        if( optionalClose ) {
            this.addPiece( new Piece.Close() );
        }
    }
    var Shape = phet.scene.Shape;
    
    Shape.prototype = {
        constructor: Shape,
        
        moveTo: function( x, y ) {
            // moveTo( point )
            if( y === undefined && x instanceof Vector2 ) {
                var point = x;
                this.addPiece( new Piece.MoveTo( point ) );
            } else { // moveTo( x, y )
                this.addPiece( new Piece.MoveTo( p( x, y ) ) );
            }
            return this;
        },
        
        lineTo: function( x, y ) {
            // lineTo( point )
            if( y === undefined && x instanceof Vector2 ) {
                var point = x;
                this.addPiece( new Piece.LineTo( point ) );
            } else { // lineTo( x, y )
                this.addPiece( new Piece.LineTo( p( x, y ) ) );
            }
            return this;
        },
        
        rect: function( a, b, c, d ) {
            // rect( upperLeft, lowerRight )
            if( c === undefined && a instanceof Vector2 && b instanceof Vector2 ) {
                this.addPiece( new Piece.Rect( a, b ) );
            } else {
                // rect( x, y, width, height )
                this.addPiece( new Piece.Rect( p( a, b ), p( a + c, b + d ) ) );
            }
            return this;
        },
        
        close: function() {
            this.addPiece( new Piece.Close() );
        },
        
        addPiece: function( piece ) {
            this.pieces.push( piece );
            piece.applyPiece( this );
            this._strokedShapeComputed = false;
        },
        
        // write out this shape's path to a canvas 2d context. does NOT include the beginPath()!
        writeToContext: function( context ) {
            _.each( this.pieces, function( piece ) {
                piece.writeToContext( context );
            } );
        },
        
        // return a new Shape that is transformed by the associated matrix
        transformed: function( matrix ) {
            return new Shape( _.flatten( _.map( this.pieces, function( piece ) { return piece.transformed( matrix ); } ), true ) );
        },
        
        // returns the bounds. if lineStyles exists, include the stroke in the bounds
        // TODO: consider renaming to getBounds()?
        computeBounds: function( lineStyles ) {
            if( lineStyles ) {
                return this.bounds.union( this.getStrokedShape( lineStyles ).bounds );
            } else {
                return this.bounds;
            }
        },
        
        getStrokedShape: function( lineStyles ) {
            if( lineStyles === undefined ) {
                lineStyles = Shape.DEFAULT_STYLES;
            }
            
            // return a cached version if possible
            if( this._strokedShapeComputed && this._strokedShapeStyles.equals( lineStyles ) ) {
                return this._strokedShape;
            }
            
            var subpaths = _.filter( this.subpaths, function( subpath ) { return subpath.isDrawable(); } );
            
            var shape = new Shape();
            
            var lineWidth = lineStyles.lineWidth;
            
            function join( center, fromTangent, toTangent ) {
                var fromPoint = center.plus( fromTangent.perpendicular().negated().times( lineWidth / 2 ) );
                var toPoint = center.plus( toTangent.perpendicular().negated().times( lineWidth / 2 ) );
                
                if( fromTangent.perpendicular().dot( toTangent ) > 0 ) {
                    switch( lineStyles.lineJoin ) {
                        case 'round':
                            throw new Error( 'stroked round lineJoin not implemented .. add arc/arcTo' );
                            break;
                        case 'miter':
                            var theta = fromTangent.angleBetween( toTangent.negated() );
                            if( 1 / Math.sin( theta / 2 ) <= lineStyles.miterLimit ) {
                                // draw the miter
                                var miterPoint = lineLineIntersection( fromPoint, fromPoint.plus( fromTangent ), toPoint, toPoint.plus( toTangent ) );
                                shape.addPiece( new Piece.LineTo( miterPoint ) );
                                shape.addPiece( new Piece.LineTo( toPoint ) );
                                break;
                            }
                            // angle too steep, use bevel instead
                        case 'bevel':
                            shape.addPiece( new Piece.LineTo( toPoint ) );
                            break;
                    }
                } else {
                    // no join necessary here since we have the acute angle. just simple lineTo for now so that the next segment starts from the right place
                    // TODO: can we prevent self-intersection here?
                    if( !fromPoint.equals( toPoint ) ) {
                        shape.addPiece( new Piece.LineTo( toPoint ) );
                    }
                }
            }
            
            function cap( center, tangent ) {
                switch( lineStyles.lineCap ) {
                    case 'butt':
                        shape.addPiece( new Piece.LineTo( center.plus( tangent.perpendicular().times( lineWidth / 2 ) ) ) );
                        break;
                    case 'round':
                        throw new Error( 'stroked round lineCap not implemented .. add arc/arcTo' );
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
                
                var alreadyClosed = _.last( segments ).end.equals( _.first( segments ).start );
                // if there is an implicit closing segment
                var closingSegment = alreadyClosed ? null : new Segment.Line( segments[segments.length-1].end, segments[0].start );
                
                // move to the first point in our stroked path
                shape.addPiece( new Piece.MoveTo( segmentStartLeft( _.first( segments ), lineWidth ) ) );
                
                // stroke the logical "left" side of our path
                for( i = 0; i < segments.length; i++ ) {
                    if( i > 0 ) {
                        join( segments[i].start, segments[i-1].endTangent, segments[i].startTangent, true );
                    }
                    shape.addPiece( segments[i].strokeLeft( lineWidth ) );
                }
                
                // handle the "endpoint"
                if( subpath.closed ) {
                    if( alreadyClosed ) {
                        join( _.last( segments ).end, _.last( segments ).endTangent, _.first( segments ).startTangent );
                        shape.addPiece( new Piece.Close() );
                        shape.addPiece( new Piece.MoveTo( segmentStartRight( _.first( segments ), lineWidth ) ) );
                        join( _.last( segments ).end, _.first( segments ).startTangent.negated(), _.last( segments ).endTangent.negated() );
                    } else {
                        join( closingSegment.start, _.last( segments ).endTangent, closingSegment.startTangent );
                        shape.addPiece( closingSegment.strokeLeft( lineWidth ) );
                        
                        // TODO: similar here to other block of if.
                        join( closingSegment.end, closingSegment.endTangent, _.first( segments ).startTangent );
                        shape.addPiece( new Piece.Close() );
                        shape.addPiece( new Piece.MoveTo( segmentStartRight( _.first( segments ), lineWidth ) ) );
                        join( closingSegment.end, _.first( segments ).startTangent.negated(), closingSegment.endTangent.negated() );
                        
                        shape.addPiece( closingSegment.strokeRight( lineWidth ) );
                        join( closingSegment.start, closingSegment.startTangent.negated(), _.last( segments ).endTangent.negated() );
                    }
                } else {
                    cap( _.last( segments ).end, _.last( segments ).endTangent );
                }
                
                // stroke the logical "right" side of our path
                for( i = segments.length - 1; i >= 0; i-- ) {
                    if( i < segments.length - 1 ) {
                        join( segments[i].end, segments[i+1].startTangent.negated(), segments[i].endTangent.negated(), false );
                    }
                    shape.addPiece( segments[i].strokeRight( lineWidth ) );
                }
                
                // handle the start point
                if( subpath.closed ) {
                    // we already did the joins, just close the 'right' side
                    shape.addPiece( new Piece.Close() );
                } else {
                    cap( _.first( segments ).start, _.first( segments ).startTangent.negated() );
                    shape.addPiece( new Piece.Close() );
                }
            } );
            
            return shape;
        },
        
        /*---------------------------------------------------------------------------*
        * Internal subpath computations
        *----------------------------------------------------------------------------*/        
        
        ensure: function( point ) {
            if( !this.hasSubpaths() ) {
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
    
    Shape.bounds = function( bounds ) {
        return new Shape().rect( p( bounds.xMin, bounds.yMin ), p( bounds.xMax, bounds.yMax ) );
    };
    
    Shape.lineSegment = function( a, b ) {
        // TODO: add type assertions?
        return new Shape().moveTo( a ).lineTo( b );
    };
    
    Shape.regularPolygon = function( sides, radius ) {
        return new Shape( _.map( _.range( sides ), function( k ) {
            var theta = 2 * Math.PI * k / sides;
            return new Piece.LineTo( p( radius * Math.cos( theta ), radius * Math.sin( theta ) ) );
        } ), true );
    };
    
    
    Shape.LineStyles = function( args ) {
        if( args === undefined ) {
            args = {};
        }
        this.lineWidth = args.lineWidth !== undefined ? args.lineWidth : 1,
        this.lineCap = args.lineCap !== undefined ? args.lineCap : 'butt', // butt, round, square
        this.lineJoin = args.lineJoin !== undefined ? args.lineJoin : 'miter', // miter, round, bevel
        this.miterLimit = args.miterLimit !== undefined ? args.miterLimit : 10 // see https://svgwg.org/svg2-draft/painting.html for miterLimit computations
    }
    var LineStyles = Shape.LineStyles;
    LineStyles.prototype = {
        constructor: LineStyles,
        
        equals: function( other ) {
            return this.lineWidth == other.lineWidth && this.lineCap == other.lineCap && this.lineJoin == other.lineJoin && this.miterLimit == other.miterLimit;
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
            this.segments.push( segment );
        },
        
        close: function() {
            this.closed = true;
        },
        
        getFirstPoint: function() {
            return _.first( this.points );
        },
        
        getLastPoint: function() {
            return _.last( this.points );
        },
        
        isDrawable: function() {
            return this.segments.length > 0;
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
            if( shape.hasSubpaths() ) {
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
            if( shape.hasSubpaths() ) {
                var start = shape.getLastSubpath().getLastPoint();
                var end = this.point;
                var line = new Segment.Line( start, end );
                shape.getLastSubpath().addSegment( line );
                shape.getLastSubpath().addPoint( end );
                shape.bounds = shape.bounds.withPoint( start ).withPoint( end );
                phet.assert( !isNaN( shape.bounds.x() ) );
            } else {
                shape.ensure( this.point );
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
            phet.assert( !isNaN( shape.bounds.x() ) );
        }
    };
    
    /*---------------------------------------------------------------------------*
    * Segments
    *----------------------------------------------------------------------------*/        
    
    Shape.Segment = {};
    var Segment = Shape.Segment;
    
    Segment.Line = function( start, end ) {
        this.start = start;
        this.end = end;
        this.startTangent = end.minus( start ).normalized();
        this.endTangent = this.startTangent;
    }
    Segment.Line.prototype = {
        constructor: Segment.Line,
        
        toPiece: function() {
            return new Piece.LineTo( this.end );
        },
        
        strokeLeft: function( lineWidth ) {
            return new Piece.LineTo( this.end.plus( this.endTangent.perpendicular().negated().times( lineWidth / 2 ) ) );
        },
        
        strokeRight: function( lineWidth ) {
            return new Piece.LineTo( this.start.plus( this.startTangent.perpendicular().times( lineWidth / 2 ) ) );
        }
    };
    
    // TODO: performance / cleanliness to have these as methods instead?
    function segmentStartLeft( segment, lineWidth ) {
        phet.assert( lineWidth !== undefined );
        return segment.start.plus( segment.startTangent.perpendicular().negated().times( lineWidth / 2 ) );
    }
    
    function segmentEndLeft( segment, lineWidth ) {
        phet.assert( lineWidth !== undefined );
        return segment.end.plus( segment.endTangent.perpendicular().negated().times( lineWidth / 2 ) );
    }
    
    function segmentStartRight( segment, lineWidth ) {
        phet.assert( lineWidth !== undefined );
        return segment.start.plus( segment.startTangent.perpendicular().times( lineWidth / 2 ) );
    }
    
    function segmentEndRight( segment, lineWidth ) {
        phet.assert( lineWidth !== undefined );
        return segment.end.plus( segment.endTangent.perpendicular().times( lineWidth / 2 ) );
    }
    
    // intersection between the line from p1-p2 and the line from p3-p4
    function lineLineIntersection( p1, p2, p3, p4 ) {
        return p(
            ( ( p1.x * p2.y - p1.y * p2.x ) * ( p3.x - p4.x ) - ( p1.x - p2.x ) * ( p3.x * p4.y - p3.y * p4.x ) ) / ( ( p1.x - p2.x ) * ( p3.y - p4.y ) - ( p1.y - p2.y ) * ( p3.x - p4.x ) ),
            ( ( p1.x * p2.y - p1.y * p2.x ) * ( p3.y - p4.y ) - ( p1.y - p2.y ) * ( p3.x * p4.y - p3.y * p4.x ) ) / ( ( p1.x - p2.x ) * ( p3.y - p4.y ) - ( p1.y - p2.y ) * ( p3.x - p4.x ) )
        );
    }
    
})();