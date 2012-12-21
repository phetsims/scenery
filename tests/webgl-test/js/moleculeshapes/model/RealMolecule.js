// Copyright 2002-2012, University of Colorado

/**
 * Represents a physically malleable version of a real molecule, with lone pairs if necessary.
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.model = phet.moleculeshapes.model || {};

// create a new scope
(function () {

    var model = phet.moleculeshapes.model;
    var Vector3 = phet.math.Vector3;

    phet.moleculeshapes.model.RealMolecule = function ( realMolecule ) {
        var i, group;

        model.Molecule.call( this );

        this.realMolecule = realMolecule;

        this.elementsUsed = {};
        this.localShapeMap = {};

        var numLonePairs = realMolecule.getCentralLonePairCount();
        var numBonds = realMolecule.getBonds().length;

        var idealCentralOrientations = [];
        var centralPairGroups = [];

        this.addCentralAtom( new model.PairGroup( new Vector3(), false, false, realMolecule.getCentralAtom().getElement() ) );

        // add in bonds
        var bonds = realMolecule.getBonds();
        for ( i = 0; i < bonds.length; i++ ) {
            var bond = bonds[i];
            var atom = bond.getOtherAtom( realMolecule.getCentralAtom() );
            var normalizedPosition = atom.position.get().normalized();
            idealCentralOrientations.push( normalizedPosition );
            var bondLength = atom.position.get().magnitude();

            var atomLocation = normalizedPosition.times( model.PairGroup.REAL_TMP_SCALE * bondLength );
            group = new model.PairGroup( atomLocation, false, false, atom.getElement() );
            centralPairGroups.add( group );
            this.addGroupAndBond( group, this.getCentralAtom(), bond.order, bondLength );
            this.elementsUsed.add( atom.getElement() );

            this.addTerminalLonePairs( group, atom.lonePairCount );
        }

        // all of the ideal vectors (including for lone pairs)
        var vseprConfiguration = new model.VseprConfiguration( numBonds, numLonePairs );
        var idealModelVectors = vseprConfiguration.getAllUnitVectors();

        var mapping = vseprConfiguration.getIdealBondRotationToPositions( model.LocalShape.sortedLonePairsFirst( this.getNeighboringAtoms( this.getCentralAtom() ) ) );

        // add in lone pairs in their correct "initial" positions
        for ( i = 0; i < numLonePairs; i++ ) {
            var normalizedPosition = mapping.rotateVector( idealModelVectors.get( i ) );
            idealCentralOrientations.add( normalizedPosition );
            group = new model.PairGroup( normalizedPosition.times( model.PairGroup.LONE_PAIR_DISTANCE ), true, false );
            this.addGroupAndBond( group, this.getCentralAtom(), 0, model.PairGroup.LONE_PAIR_DISTANCE / model.PairGroup.REAL_TMP_SCALE );
            centralPairGroups.add( group );
        }

        this.localShapeMap[this.getCentralAtom().id] = new model.LocalShape( model.LocalShape.realPermutations( centralPairGroups ), this.getCentralAtom(), centralPairGroups, idealCentralOrientations );

        // basically only use VSEPR model for the attraction on non-central atoms
        var radialAtoms = this.getRadialAtoms();
        for ( i = 0; i < radialAtoms.length; i++ ) {
            this.localShapeMap[radialAtoms[i].id] = this.getLocalVSEPRShape( radialAtoms[i] );
        }
    };

    // essentially inheritance
    var RealMolecule = phet.moleculeshapes.model.RealMolecule;
    RealMolecule.prototype = Object.create( model.Molecule.prototype );
    RealMolecule.prototype.constructor = RealMolecule;

    RealMolecule.prototype.update = function ( tpf ) {
        model.Molecule.prototype.update.call( this, tpf );

        // angle-based repulsion
        var atoms = this.getAtoms();
        for ( var i = 0; i < atoms.length; i++ ) {
            var atom = atoms[i];
            var neighbors = this.getNeighbors( atom );
            if ( neighbors.length > 1 ) {
                var localShape = this.getLocalShape( atom );

                localShape.applyAngleAttractionRepulsion( tpf );
            }
        }
    };

    RealMolecule.prototype.getLocalShape = function ( atom ) {
        return this.localShapeMap[atom.id];
    };

    RealMolecule.prototype.isReal = true;

    RealMolecule.prototype.getMaximumBondLength = function () {
        return undefined;
    };

    RealMolecule.prototype.getRealMolecule = function () {
        return this.realMolecule;
    };
})();