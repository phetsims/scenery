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
