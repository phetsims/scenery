// Copyright 2002-2012, University of Colorado

/**
 * An immutable rectangle-shaped bounded area (bounding box) in 2D
 *
 * @author Jonathan Olson
 */

// ensure proper namespace
var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {
    var Vector2 = phet.math.Vector2;
    
    // not using x,y,width,height so that it can handle infinity-based cases in a better way
    phet.math.Bounds2 = function( xMin, yMin, xMax, yMax ) {
        this.xMin = xMin;
        this.yMin = yMin;
        this.xMax = xMax;
        this.yMax = yMax;
    };

    var Bounds2 = phet.math.Bounds2;

    Bounds2.prototype = {
        constructor: Bounds2,
        
        // properties of this bounding box
        width: function() { return this.xMax - this.xMin; },
        height: function() { return this.yMax - this.yMin; },
        x: function() { return this.xMin; },
        y: function() { return this.yMin; },
        centerX: function() { return ( this.xMax + this.xMin ) / 2; },
        centerY: function() { return ( this.yMax + this.yMin ) / 2; },
        isEmpty: function() { return this.width() <= 0 || this.height() <= 0; },
        
        // immutable operations (bounding-box style handling, so that the relevant bounds contain everything)
        union: function( other ) {
            return new Bounds2(
                Math.min( this.xMin, other.xMin ),
                Math.min( this.yMin, other.yMin ),
                Math.max( this.xMax, other.xMax ),
                Math.max( this.yMax, other.yMax )
            );
        },
        intersection: function( other ) {
            return new Bounds2(
                Math.max( this.xMin, other.xMin ),
                Math.max( this.yMin, other.yMin ),
                Math.min( this.xMax, other.xMax ),
                Math.min( this.yMax, other.yMax )
            );
        },
        // TODO: difference should be well-defined, but more logic is needed to compute
        
        // like a union with a point-sized bounding box
        withPoint: function( point ) {
            return new Bounds2(
                Math.min( this.xMin, point.x ),
                Math.min( this.yMin, point.y ),
                Math.max( this.xMax, point.x ),
                Math.max( this.yMax, point.y )
            );
        },
        
        withMinX: function( minX ) { return new Bounds2( minX, this.yMin, this.xMax, this.yMax ); },
        withMinY: function( minY ) { return new Bounds2( this.xMin, minY, this.xMax, this.yMax ); },
        withMaxX: function( maxX ) { return new Bounds2( this.xMin, this.yMin, maxX, this.yMax ); },
        withMaxY: function( maxY ) { return new Bounds2( this.xMin, this.yMin, this.xMax, maxY ); },
        
        // copy rounded to integral values, expanding where necessary
        roundedOut: function() {
            return new Bounds2(
                Math.floor( this.xMin ),
                Math.floor( this.yMin ),
                Math.ceil( this.xMax ),
                Math.ceil( this.yMax )
            );
        },
        
        // copy rounded to integral values, contracting where necessary
        roundedIn: function() {
            return new Bounds2(
                Math.ceil( this.xMin ),
                Math.ceil( this.yMin ),
                Math.floor( this.xMax ),
                Math.floor( this.yMax )
            );
        },
        
        // whether the point is inside the bounding box
        containsPoint: function( point ) {
            return this.xMin <= point.x && point.x <= this.xMax && this.yMin <= point.y && point.y <= this.yMax;
        },
        
        // whether this bounding box completely contains the argument bounding box
        containsBounds: function( bounds ) {
            return this.xMin <= bounds.xMin && this.xMax >= bounds.xMax && this.yMin <= bounds.yMin && this.yMax >= bounds.yMax;
        },
        
        intersectsBounds: function( bounds ) {
            // TODO: more efficient way of doing this?
            return !this.intersection( bounds ).isEmpty();
        },
        
        // transform a bounding box.
        // NOTE that box.transformed( matrix ).transformed( inverse ) may be larger than the original box
        transformed: function( matrix ) {
            if( this.isEmpty() ) {
                return Bounds2.NOTHING;
            }
            var result = Bounds2.NOTHING;
            
            // make sure all 4 corners are inside this transformed bounding box
            result = result.withPoint( matrix.timesVector2( new Vector2( this.xMin, this.yMin ) ) );
            result = result.withPoint( matrix.timesVector2( new Vector2( this.xMin, this.yMax ) ) );
            result = result.withPoint( matrix.timesVector2( new Vector2( this.xMax, this.yMin ) ) );
            result = result.withPoint( matrix.timesVector2( new Vector2( this.xMax, this.yMax ) ) );
            return result;
        },
        
        // returns copy expanded on all sides by length d
        dilated: function( d ) {
            return new Bounds2( this.xMin - d, this.yMin - d, this.xMax + d, this.yMax + d );
        },
        
        // returns copy contracted on all sides by length d
        eroded: function( d ) {
            return this.dilated( -d );
        },

        toString: function () {
            return '[x:(' + this.xMin + ',' + this.xMax + '),y:(' + this.yMin + ',' + this.yMax + ')]';
        },

        equals: function ( other ) {
            return this.xMin === other.xMin && this.yMin === other.yMin && this.xMax === other.xMax && this.yMax === other.yMax;
        }
    };
    
    // specific bounds useful for operations
    Bounds2.EVERYTHING = new Bounds2( Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY );
    Bounds2.NOTHING = new Bounds2( Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY );
})();
