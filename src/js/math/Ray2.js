// Copyright 2002-2012, University of Colorado

/**
 * 2-dimensional ray
 *
 * @author Jonathan Olson
 */

// ensure proper namespace
var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {
    "use strict";

    phet.math.Ray2 = function ( pos, dir ) {
        this.pos = pos;
        this.dir = dir;
    };

    var Ray2 = phet.math.Ray2;

    Ray2.prototype = {
        constructor: Ray2,

        shifted: function ( distance ) {
            return new Ray2( this.pointAtDistance( distance ), this.dir );
        },

        pointAtDistance: function ( distance ) {
            return this.pos.plus( this.dir.timesScalar( distance ) );
        },

        toString: function () {
            return this.pos.toString() + " => " + this.dir.toString();
        }
    };
})();
