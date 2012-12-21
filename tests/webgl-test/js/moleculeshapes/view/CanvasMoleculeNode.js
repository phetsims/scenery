// Copyright 2002-2012, University of Colorado

/**
 * Displays a 3D Molecule
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.view = phet.moleculeshapes.view || {};

// create a new scope
(function () {

    phet.moleculeshapes.view.CanvasMoleculeNode = function ( context, molecule ) {
        phet.webgl.GLNode.call( this );

        this.context = context;
        this.molecule = molecule;
    };

    var CanvasMoleculeNode = phet.moleculeshapes.view.CanvasMoleculeNode;

    CanvasMoleculeNode.prototype = Object.create( phet.webgl.GLNode.prototype );
    CanvasMoleculeNode.prototype.constructor = CanvasMoleculeNode;
})();