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
    
    phet.scene.Shape = function() {
        this.pieces = [];
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
    
    // TODO: convenience functions for other types
    
    Shape.prototype = {
        constructor: Shape,
        
        addPiece: function( piece ) {
            this.pieces.push( piece );
        },
        
        // write out this shape's path to a canvas 2d context
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
        }
    };
})();