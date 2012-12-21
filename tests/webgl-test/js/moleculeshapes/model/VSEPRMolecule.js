// Copyright 2002-2012, University of Colorado

/**
 * A molecule that behaves with a behavior that doesn't discriminate between bond or atom types (only lone pairs vs bonds)
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.model = phet.moleculeshapes.model || {};

// create a new scope
(function () {

    var model = phet.moleculeshapes.model;

    phet.moleculeshapes.model.VSEPRMolecule = function ( bondLengthOverride ) {
        model.Molecule.call( this );

        this.bondLengthOverride = bondLengthOverride;
    };

    // essentially inheritance
    var VSEPRMolecule = phet.moleculeshapes.model.VSEPRMolecule;
    VSEPRMolecule.prototype = Object.create( model.Molecule.prototype );
    VSEPRMolecule.prototype.constructor = VSEPRMolecule;

    VSEPRMolecule.prototype.update = function ( tpf ) {
        model.Molecule.prototype.update.call( this, tpf );

        var radialGroups = this.getRadialGroups();

        var atoms = this.getAtoms();
        for ( var i = 0; i < atoms.length; i++ ) {
            var atom = atoms[i];
            if ( this.getNeighbors( atom ).length > 1 ) {
                if ( atom.isCentralAtom() ) {
                    // attractive force to the correct position
                    var error = this.getLocalShape( atom ).applyAttraction( tpf );

                    // factor that basically states "if we are close to an ideal state, force the coulomb force to ignore differences between bonds and lone pairs based on their distance"
                    var trueLengthsRatioOverride = Math.max( 0, Math.min( 1, Math.log( error + 1 ) - 0.5 ) );

                    for ( var j = 0; j < radialGroups.length; j++ ) {
                        var group = radialGroups[j];
                        for ( var k = 0; k < radialGroups.length; k++ ) {
                            var otherGroup = radialGroups[k];

                            if ( otherGroup != group && group != this.getCentralAtom() ) {
                                group.repulseFrom( otherGroup, tpf, trueLengthsRatioOverride );
                            }
                        }
                    }
                }
                else {
                    // handle terminal lone pairs gracefully
                    this.getLocalShape( atom ).applyAngleAttractionRepulsion( tpf );
                }
            }
        }
    };

    VSEPRMolecule.prototype.getLocalShape = function ( atom ) {
        return this.getLocalVSEPRShape( atom );
    };

    VSEPRMolecule.prototype.isReal = false;

    VSEPRMolecule.prototype.getMaximumBondLength = function () {
        if ( this.bondLengthOverride !== undefined ) {
            return this.bondLengthOverride;
        }
        else {
            return model.PairGroup.BONDED_PAIR_DISTANCE;
        }
    };
})();