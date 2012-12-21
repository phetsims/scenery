// Copyright 2002-2012, University of Colorado

/**
 * Bond between atoms
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.model = phet.moleculeshapes.model || {};

// create a new scope
(function () {

    phet.moleculeshapes.model.Bond = function ( a, b, order, length ) {
        this.a = a;
        this.b = b;
        this.order = order;
        this.length = length; // in angstroms, or 0 / undefined if there is no data
    };

    var Bond = phet.moleculeshapes.model.Bond;

    Bond.prototype = {
        constructor: Bond,

        toString: function () {
            return "{" + this.a.toString() + " => " + this.b.toString() + "}";
        },

        contains: function ( atom ) {
            return this.a === atom || this.b === atom;
        },

        getOtherAtom: function ( atom ) {
            phet.assert( this.contains( atom ) );

            return this.a === atom ? this.b : this.a;
        },

        equals: function ( bond ) {
            // TODO: consider checking bond order? or is this not important?
            return this.a === bond.a && this.b === bond.b;
        }
    }
})();