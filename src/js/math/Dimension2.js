// Copyright 2002-2012, University of Colorado

/**
 * Basic width and height
 *
 * @author Jonathan Olson
 */

// ensure proper namespace
var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {
    phet.math.Dimension2 = function ( width, height ) {
        this.width = width;
        this.height = height;
    };

    var Dimension2 = phet.math.Dimension2;

    Dimension2.prototype = {
        constructor: Dimension2,

        toString: function () {
            return "[" + this.width + "w, " + this.height + "h]";
        },

        equals: function ( other ) {
            return this.width === other.width && this.height === other.height;
        }
    };
})();
