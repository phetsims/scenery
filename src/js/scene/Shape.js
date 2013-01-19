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
    
    Shape.LineStyles = function( args ) {
        if( args === undefined ) {
            args = {};
        }
        return {
            lineWidth: args.lineWidth !== undefined ? args.lineWidth : 1,
            lineCap: args.lineCap !== undefined ? args.lineCap : 'butt', // butt, round, square
            lineJoin: args.lineJoin !== undefined ? args.lineJoin : 'miter', // miter, round, bevel
            miterLimit: args.miterLimit !== undefined ? args.miterLimit : 10 // see https://svgwg.org/svg2-draft/painting.html for miterLimit computations
        };
    }
    var LineStyles = Shape.LineStyles;
    
    // Pieces should be considered immutable
    Shape.Piece = function( points ) {
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
        Piece.call( this, [] );
    };
    Piece.Close.prototype = Object.create( Piece.prototype );
    Piece.Close.prototype.constructor = Piece.Close;
    Piece.Close.prototype.writeToContext = function( context ) { context.closePath(); };
    
    Piece.MoveTo = function( point ) {
        Piece.call( this, [ point ] );
        this.point = point;
    };
    Piece.MoveTo.prototype = Object.create( Piece.prototype );
    Piece.MoveTo.prototype.constructor = Piece.MoveTo;
    Piece.MoveTo.prototype.writeToContext = function( context ) { context.moveTo( this.point.x, this.point.y ); };
    
    Piece.LineTo = function( point ) {
        Piece.call( this, [ point ] );
        this.point = point;
    };
    Piece.LineTo.prototype = Object.create( Piece.prototype );
    Piece.LineTo.prototype.constructor = Piece.LineTo;
    Piece.LineTo.prototype.writeToContext = function( context ) { context.lineTo( this.point.x, this.point.y ); };
    
    // TODO: transforms for rect are different!!!
    Piece.Rect = function( upperLeft, lowerRight ) {
        Piece.call( this, [ upperLeft, lowerRight ] );
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
            } );
        },
        
        // return a new Shape that is transformed by the associated matrix
        transformed: function( matrix ) {
            // TODO: reimplement transformations
            return this;
        },
        
        // returns the bounds. if lineDrawingStyles exists, include the stroke in the bounds
        computeBounds: function( lineDrawingStyles ) {
            
            var shapeState = new ShapeState();
            
            // TODO: include stroke information somehow
            _.each( this.pieces, function( piece ) {
                piece.applyPiece( shapeState );
            } );
            
            return shapeState.bounds;
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