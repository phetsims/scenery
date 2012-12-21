// Copyright 2002-2012, University of Colorado

/**
 * An atom with a 3D position
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.model = phet.moleculeshapes.model || {};

// create a new scope
(function () {
    var Property = phet.model.Property;

    phet.moleculeshapes.model.Atom = function ( element, position, lonePairCount ) {
        phet.chemistry.Atom.call( this, element );

        this.position = new Property( position );
        this.lonePairCount = lonePairCount || 0;
    };

    var Atom = phet.moleculeshapes.model.Atom;

    Atom.prototype = Object.create( phet.chemistry.Atom.prototype );
    Atom.prototype.constructor = Atom;
})();