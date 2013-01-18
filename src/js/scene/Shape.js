// Copyright 2002-2012, University of Colorado

/**
 * Shape handling
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    
    var Vector2 = phet.math.Vector2;
    
    // for brevity
    function p( x,y ) {
        return new Vector2( x, y );
    }
    
    phet.scene.Shape = function( pieces, optionalClose ) {
        this.pieces = pieces !== undefined ? pieces : [];
        if( optionalClose ) {
            this.addPiece( new Piece.Close() );
        }
    }
    
    var Shape = phet.scene.Shape;
    
    // winding rule constants
    Shape.FILL_NONZERO = 0;
    Shape.FILL_EVENODD = 1;
    
    // piece types
    Shape.PIECE_MOVE = 0;
    Shape.PIECE_LINE = 1;
    Shape.PIECE_CLOSE = 2;
    Shape.PIECE_QUADRATIC = 3;
    Shape.PIECE_CUBIC = 4;
    Shape.PIECE_ELLIPSE = 5;
    Shape.PIECE_ARC_TO = 6;
    Shape.PIECE_ARC = 7;
    Shape.PIECE_RECT = 8;
    // TODO: simplify arc/ellipse/cubic/quadratic parts?
    
    // line caps
    Shape.CAP_BUTT = 0;
    Shape.CAP_ROUND = 1;
    Shape.CAP_SQUARE = 2;
    
    // line joins
    Shape.JOIN_ROUND = 0;
    Shape.JOIN_BEVEL = 1;
    Shape.JOIN_MITER = 2;
    
    Shape.LineStyles = function( args ) {
        if( args === undefined ) {
            args = {};
        }
        return {
            lineWidth: args.lineWidth !== undefined ? args.lineWidth : 1,
            lineCap: args.lineCap !== undefined ? args.lineCap : Shape.CAP_BUTT,
            lineJoin: args.lineJoin !== undefined ? args.lineJoin : Shape.JOIN_MITER,
            miterLimit: args.miterLimit !== undefined ? args.miterLimit : 10 // see https://svgwg.org/svg2-draft/painting.html for miterLimit computations
        };
    }
    
    // default canvas styles according to spec
    Shape.DEFAULT_STYLES = new Shape.LineStyles();
    
    // Pieces should be considered immutable
    Shape.Piece = function( type, points ) {
        this.type = type;
        this.points = points;
    };
    
    var Piece = Shape.Piece;
    
    Piece.prototype = {
        constructor: Piece,
        
        // override for more specific transforms. 
        transformed: function( matrix ) {
            // TODO: implement transformations! -- and verify with unit tests
            return this;
        },
        
        applyPiece: function( shapeState ) {
            _.each( this.points, function( point ) {
                shapeState.bounds = shapeState.bounds.withPoint( point );
            } );
        }
    };
    
    // TODO: make a canonical CLOSE?
    Piece.Close = function() {
        Piece.call( this, Shape.PIECE_CLOSE, [] );
    };
    Piece.Close.prototype = Object.create( Piece.prototype );
    Piece.Close.prototype.constructor = Piece.Close;
    Piece.Close.prototype.writeToContext = function( context ) { context.closePath(); };
    
    Piece.MoveTo = function( point ) {
        Piece.call( this, Shape.PIECE_MOVE, [ point ] );
        this.point = point;
    };
    Piece.MoveTo.prototype = Object.create( Piece.prototype );
    Piece.MoveTo.prototype.constructor = Piece.MoveTo;
    Piece.MoveTo.prototype.writeToContext = function( context ) { context.moveTo( this.point.x, this.point.y ); };
    
    Piece.LineTo = function( point ) {
        Piece.call( this, Shape.PIECE_LINE, [ point ] );
        this.point = point;
    };
    Piece.LineTo.prototype = Object.create( Piece.prototype );
    Piece.LineTo.prototype.constructor = Piece.LineTo;
    Piece.LineTo.prototype.writeToContext = function( context ) { context.lineTo( this.point.x, this.point.y ); };
    
    // TODO: transforms for rect are different!!!
    Piece.Rect = function( upperLeft, lowerRight ) {
        Piece.call( this, Shape.PIECE_RECT, [ upperLeft, lowerRight ] );
        this.upperLeft = upperLeft;
        this.lowerRight = lowerRight;
        this.x = this.upperLeft.x;
        this.y = this.upperLeft.y;
        this.width = this.lowerRight.x - this.x;
        this.height = this.lowerRight.y - this.y;
    };
    Piece.Rect.prototype = Object.create( Piece.prototype );
    Piece.Rect.prototype.constructor = Piece.Rect;
    Piece.Rect.prototype.writeToContext = function( context ) { context.rect( this.x, this.y, this.width, this.height ); };
    
    
    Shape.ShapeState = function() {
        this.subpaths = [];
        this.bounds = phet.math.Bounds2.NOTHING;
    };
    
    var ShapeState = Shape.ShapeState;
    
    ShapeState.prototype = {
        constructor: ShapeState,
        
        ensure: function( point ) {
            
        }
    };
    
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
        
        
        addPiece: function( piece ) {
            phet.assert( piece.writeToContext );
            this.pieces.push( piece );
        },
        
        // write out this shape's path to a canvas 2d context. does NOT include the beginPath()!
        writeToContext: function( context ) {
            _.each( this.pieces, function( piece ) {
                if( !piece.writeToContext ) {
                    console.log( piece );
                }
                piece.writeToContext( context );
                // switch( piece.type ) {
                //     case Shape.PIECE_MOVE: context.moveTo( piece.points[0].x, piece.points[0].y ); break;
                //     case Shape.PIECE_LINE: context.lineTo( piece.points[0].x, piece.points[0].y ); break;
                //     case Shape.PIECE_CLOSE: context.closePath(); break;
                //     case Shape.PIECE_QUADRATIC: context.quadraticCurveTo( piece.points[0].x, piece.points[0].y, piece.points[1].x, piece.points[1].y ); break;
                //     case Shape.PIECE_CUBIC: context.bezierCurveTo( piece.points[0].x, piece.points[0].y, piece.points[1].x, piece.points[1].y, piece.points[2].x, piece.points[2].y ); break;
                //     case Shape.PIECE_ELLIPSE: context.ellipse( piece.points[0].x, piece.points[0].y, piece.args.radiusX, piece.args.radiusY, piece.args.rotation, piece.args.startAngle, piece.args.endAngle, piece.args.anticlockwise ); break;
                //     case Shape.PIECE_ARC_TO:
                //         // TODO: consider splitting ARC_TO into two separate piece types
                //         if( piece.args.radius !== undefined ) {
                //             context.ellipse( piece.points[0].x, piece.points[0].y, piece.points[1].x, piece.points[1].y, piece.args.radius );
                //         } else {
                //             context.ellipse( piece.points[0].x, piece.points[0].y, piece.points[1].x, piece.points[1].y, piece.args.radiusX, piece.args.radiusY, piece.args.rotation );
                //         }
                //         break;
                //     case Shape.PIECE_ARC: context.arc( piece.points[0].x, piece.points[0].y, piece.args.radius, piece.args.startAngle, piece.args.endAngle, piece.args.anticlockwise ); break;
                //     case Shape.PIECE_RECT: context.rect( piece.points[0].x, piece.points[0].y, piece.points[1].x - piece.points[0].x, piece.points[1].y - piece.points[0].y ); break;
                //     default:
                //         throw new Error( 'writeToContext unimplemented for piece type' + piece.type );
                // }
            } );
        },
        
        // return a new Shape that is transformed by the associated matrix
        transformed: function( matrix ) {
            // TODO: reimplement transformations
            return this;
            /*
            return new Shape( _.map( this.pieces, function( piece ) {
                var transformedPoints = _.map( piece.points, function( point ) { return matrix.timesVector2( point ); } );
                var args = piece.args;
                switch( piece.type ) {
                    case Shape.PIECE_MOVE:
                    case Shape.PIECE_LINE:
                    case Shape.PIECE_CLOSE:
                    case Shape.PIECE_QUADRATIC:
                    case Shape.PIECE_CUBIC:
                    case Shape.PIECE_RECT:
                        return new Piece( piece.type, transformedPoints, args );
                    case Shape.PIECE_ELLIPSE:
                        return new Piece( Shape.PIECE_ELLIPSE, transformedPoints, {
                            // TODO: more convenient way of handling one-param modifications, like args.with( { rotation: <blah> } )
                            radiusX: args.radiusX,
                            radiusY: args.radiusY,
                            rotation: args.rotation + matrix.rotation(),
                            startAngle: args.startAngle,
                            endAngle: args.endAngle,
                            anticlockwise: args.anticlockwise
                        } );
                    case Shape.PIECE_ARC_TO:
                        if( args.radius !== undefined ) {
                            return new Piece( Shape.PIECE_ARC_TO, transformedPoints, args );
                        } else {
                            return new Piece( Shape.PIECE_ARC_TO, transformedPoints, {
                                radiusX: args.radiusX,
                                radiusY: args.radiusY,
                                rotation: args.rotation + matrix.rotation()
                            } );
                        }
                    case Shape.PIECE_ARC:
                        var extraRotation = matrix.rotation();
                        return new Piece( Shape.PIECE_ARC, transformedPoints, {
                            radius: args.radius,
                            startAngle: args.startAngle + extraRotation,
                            endAngle: args.endAngle + extraRotation,
                            anticlockwise: args.anticlockwise
                        } );
                    default:
                        throw new Error( 'transformed unimplemented for piece type' + piece.type );
                }
            } ));
            */
        },
        
        // returns the bounds. 
        computeBounds: function( lineDrawingStyles ) {
            
            // TODO: consider null => no stroke?
            if( lineDrawingStyles === undefined ) {
                lineDrawingStyles = Shape.DEFAULT_STYLES;
            }
            
            var shapeState = new ShapeState();
            
            _.each( this.pieces, function( piece ) {
                piece.applyPiece( shapeState );
            } );
            
            return shapeState.bounds;
            
            /*
            var bounds = phet.math.Bounds2.NOTHING;
            
            // TODO: improve bounds constraints (not as tight as possible yet)
            _.each( this.pieces, function( piece ) {
                // set bounding box to contain all control points
                _.each( piece.points, function( point ) {
                    bounds = bounds.withPoint( point );
                } );
                
                switch( piece.type ) {
                    case Shape.PIECE_MOVE:
                    case Shape.PIECE_LINE:
                    case Shape.PIECE_CLOSE:
                    case Shape.PIECE_QUADRATIC:
                    case Shape.PIECE_CUBIC:
                    case Shape.PIECE_RECT:
                        break; // already handled by the points. TODO implement tighter on curves due to control points not being bounds
                    case Shape.PIECE_ELLIPSE:
                        var x = piece.points[0].x;
                        var y = piece.points[0].y;
                        var maxRadius = Math.max( piece.args.radiusX, piece.args.radiusY );
                        bounds = bounds.union( new phet.math.Bounds2( x - maxRadius, y - maxRadius, x + maxRadius, y - maxRadius ) );
                        break;
                    case Shape.PIECE_ARC_TO:
                        throw new Error( 'arcTo computeBounds not implemented yet' );
                    case Shape.PIECE_ARC:
                        var x = piece.points[0].x;
                        var y = piece.points[0].y;
                        var radius = piece.args.radius;
                        bounds = bounds.union( new phet.math.Bounds2( x - radius, y - radius, x + radius, y - radius ) );
                        break;
                    default:
                        throw new Error( 'computeBounds unimplemented for piece type: ' + piece.type );
                }
            } );
            return bounds;
            */
        },
        
        traced: function( lineDrawingStyles ) {
            if( lineDrawingStyles === undefined ) {
                lineDrawingStyles = Shape.DEFAULT_STYLES;
            }
            
            var path = [];
            var subpath = [];
            
            // TODO: return a shape that is this current shape's traced form with a stroke. see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#trace-a-path
            throw new Error( 'Shape traced unimplemented' );
        }
    };

    
})();