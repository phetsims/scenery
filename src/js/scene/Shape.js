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
    
    phet.scene.Shape = function( pieces ) {
        this.pieces = pieces !== undefined ? pieces : [];
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
    Shape.Piece = function( type, points, args ) {
        this.type = type;
        this.points = points;
        this.args = args;
    };
    
    var Piece = Shape.Piece;
    
    Piece.moveTo = function( x, y ) {
        return Piece.moveToPoint( new Vector2( x, y ) );
    };
    
    Piece.moveToPoint = function( point ) {
        return new Piece( Shape.PIECE_MOVE, [ point ] );
    };
    
    Piece.lineTo = function( x, y ) {
        return Piece.lineToPoint( new Vector2( x, y ) );
    };
    
    Piece.lineToPoint = function( point ) {
        return new Piece( Shape.PIECE_LINE, [ point ] );
    };
    
    Piece.closePath = function() {
        return new Piece( Shape.PIECE_CLOSE, [] );
    };
    
    Piece.rect = function( x, y, width, height ) {
        return new Piece( Shape.PIECE_RECT, [ new Vector2( x, y ), new Vector2( x + width, y + height ) ] );
    };
    
    // TODO: convenience functions for other piece types
    
    
    Shape.rectangle = function( x, y, width, height ) {
        return new Shape( [ Piece.rect( x, y, width, height ) ] );
    };
    
    
    Shape.prototype = {
        constructor: Shape,
        
        addPiece: function( piece ) {
            this.pieces.push( piece );
        },
        
        // write out this shape's path to a canvas 2d context. does NOT include the beginPath()!
        writeToContext: function( context ) {
            _.each( this.pieces, function( piece ) {
                switch( piece.type ) {
                    case Shape.PIECE_MOVE: context.moveTo( piece.points[0].x, piece.points[0].y ); break;
                    case Shape.PIECE_LINE: context.lineTo( piece.points[0].x, piece.points[0].y ); break;
                    case Shape.PIECE_CLOSE: context.closePath(); break;
                    case Shape.PIECE_QUADRATIC: context.quadraticCurveTo( piece.points[0].x, piece.points[0].y, piece.points[1].x, piece.points[1].y ); break;
                    case Shape.PIECE_CUBIC: context.bezierCurveTo( piece.points[0].x, piece.points[0].y, piece.points[1].x, piece.points[1].y, piece.points[2].x, piece.points[2].y ); break;
                    case Shape.PIECE_ELLIPSE: context.ellipse( piece.points[0].x, piece.points[0].y, piece.args.radiusX, piece.args.radiusY, piece.args.rotation, piece.args.startAngle, piece.args.endAngle, piece.args.anticlockwise ); break;
                    case Shape.PIECE_ARC_TO:
                        // TODO: consider splitting ARC_TO into two separate piece types
                        if( piece.args.radius !== undefined ) {
                            context.ellipse( piece.points[0].x, piece.points[0].y, piece.points[1].x, piece.points[1].y, piece.args.radius );
                        } else {
                            context.ellipse( piece.points[0].x, piece.points[0].y, piece.points[1].x, piece.points[1].y, piece.args.radiusX, piece.args.radiusY, piece.args.rotation );
                        }
                        break;
                    case Shape.PIECE_ARC: context.arc( piece.points[0].x, piece.points[0].y, piece.args.radius, piece.args.startAngle, piece.args.endAngle, piece.args.anticlockwise ); break;
                    case Shape.PIECE_RECT: context.rect( piece.points[0].x, piece.points[0].y, piece.points[1].x - piece.points[0].x, piece.points[1].y - piece.points[0].y ); break;
                }
            } );
        },
        
        decompose: function() {
            // TODO: will return a Shape using simple piece types
        },
        
        // return a new Shape that is transformed by the associated matrix
        transformed: function( matrix ) {
            return new Shape( _.map( this.pieces, function( piece ) {
                var transformedPoints = _.map( piece.points, matrix.timesVector2 );
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
                        break;
                    case Shape.PIECE_ARC:
                        var extraRotation = matrix.rotation();
                        return new Piece( Shape.PIECE_ARC, transformedPoints, {
                            radius: args.radius,
                            startAngle: args.startAngle + extraRotation,
                            endAngle: args.endAngle + extraRotation,
                            anticlockwise: args.anticlockwise
                        } );
                }
            } ));
        },
        
        // returns the bounds. 
        computeBounds: function( lineDrawingStyles ) {
            
            // TODO: consider null => no stroke?
            if( lineDrawingStyles === undefined ) {
                lineDrawingStyles = Shape.DEFAULT_STYLES;
            }
            
            // TODO: return Bounds2
            return phet.math.Bounds2.NOTHING;
        },
        
        traced: function( lineDrawingStyles ) {
            // TODO: return a shape that is this current shape's traced form with a stroke. see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#trace-a-path
        }
    };
    
})();