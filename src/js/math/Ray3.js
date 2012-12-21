// Copyright 2002-2012, University of Colorado

/**
 * 3-dimensional ray
 *
 * @author Jonathan Olson
 */

// ensure proper namespace
var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {

    phet.math.Ray3 = function ( pos, dir ) {
        this.pos = pos;
        this.dir = dir;
    };

    var Ray3 = phet.math.Ray3;

    Ray3.prototype = {
        constructor: Ray3,

        shifted: function ( distance ) {
            return new Ray3( this.pointAtDistance( distance ), this.dir );
        },

        pointAtDistance: function ( distance ) {
            return this.pos.plus( this.dir.timesScalar( distance ) );
        },

        toString: function () {
            return this.pos.toString() + " => " + this.dir.toString();
        }
    };
})();
