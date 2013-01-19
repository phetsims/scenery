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
        
        // flag to see whether we have 
        this.computed = false;
        
        this.subpaths = [];
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
        
        applyPiece: function( shapeState ) {
            
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
        
        applyPiece: function( shapeState ) {
            // TODO: technically, nothing here! - moveTos in a row could cause problems
            shapeState.bounds = shapeState.bounds.withPoint( this.point );
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
        
        applyPiece: function( shapeState ) {
            // TODO: start and end points!
            shapeState.bounds = shapeState.bounds.withPoint( this.point );
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
        
        applyPiece: function( shapeState ) {
            shapeState.bounds = shapeState.bounds.withPoint( this.upperLeft ).withPoint( this.lowerRight );
        }
    };
    
    
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