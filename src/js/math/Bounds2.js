// Copyright 2002-2012, University of Colorado

/**
 * An immutable rectangle-shaped bounded area in 2D
 *
 * @author Jonathan Olson
 */

// ensure proper namespace
var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {
    phet.math.Bounds2 = function ( x, y, width, height ) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    };

    // TODO: min/max/center values

    var Bounds2 = phet.math.Bounds2;

    Bounds2.prototype = {
        constructor: Bounds2,

        toString: function () {
            return "[(" + this.x + "," + this.y + ") " + this.width + "w, " + this.height + "h]";
        },

        equals: function ( other ) {
            return this.x === other.x && this.y === other.y && this.width === other.width && this.height === other.height;
        }
    };
})();
