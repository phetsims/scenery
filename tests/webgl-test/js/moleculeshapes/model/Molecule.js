// Copyright 2002-2012, University of Colorado

/**
 * Model of a single-atom-centered molecule which has a certain number of pair groups
 * surrounding it.
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.model = phet.moleculeshapes.model || {};

// create a new scope
(function () {

    var model = phet.moleculeshapes.model;
    var Notifier = phet.model.Notifier;
    var CompositeNotifier = phet.model.CompositeNotifier;

    var MAX_PAIRS = 6;

    phet.moleculeshapes.model.Molecule = function () {
        // all of the pair groups
        this.groups = [];

        // bonds between pair groups. for lone pairs, this doesn't mean an actual molecular bond, so we just have order 0
        this.bonds = [];

        this.centralAtom = null; // will be filled in later

        this.onBondAdded = new Notifier();
        this.onBondRemoved = new Notifier();
        this.onBondChanged = new CompositeNotifier( [this.onBondAdded, this.onBondRemoved] );

        this.onGroupAdded = new Notifier();
        this.onGroupRemoved = new Notifier();
        this.onGroupChanged = new CompositeNotifier( [this.onGroupAdded, this.onGroupRemoved] );
    };

    var Molecule = phet.moleculeshapes.model.Molecule;
    var map = phet.util.map;
    var filter = phet.util.filter;

    Molecule.prototype = {
        constructor: Molecule,

        // abstract getLocalShape( atom )
        // abstract getMaximumBondLength() -- option
        // abstract isReal()

        update: function ( tpf ) {
            var that = this;
            var nonCentralGroups = phet.util.filter( this.groups, function ( group ) {return group != that.centralAtom;} );

            // move based on velocity
            for ( var i = 0; i < nonCentralGroups.length; i++ ) {
                var group = nonCentralGroups[i];

                var parentBond = this.getParentBond( group );
                var origin = parentBond.getOtherAtom( group ).position.get();

                var oldDistance = ( group.position.get().minus( origin ) ).magnitude();
                group.stepForward( tpf );
                group.attractToIdealDistance( tpf, oldDistance, parentBond );
            }
        },

        getAtoms: function () {
            return filter( this.groups, function ( group ) {return !group.isLonePair;} );
        },

        // the number of surrounding pair groups
        getStericNumber: function ( group ) {
            return this.getBonds( group ).length;
        },

        getBonds: function ( group ) {
            if ( group ) {
                // all bonds to the pair group, if specified
                return filter( this.bonds, function ( bond ) {return bond.contains( group )} );
            }
            else {
                return this.bonds;
            }
        },

        // all neighboring pair groups
        getNeighbors: function ( group ) {
            return map( this.getBonds( group ), function ( bond ) {return bond.getOtherAtom( group )} );
        },

        getAllNonCentralAtoms: function () {
            var centralAtom = this.centralAtom;
            return filter( this.groups, function ( group ) { return !group.isLonePair && group != centralAtom; } );
        },

        getAllLonePairs: function () {
            return filter( groups, function ( group ) {return group.isLonePair;} );
        },

        // atoms surrounding the center atom
        getRadialAtoms: function () {
            return this.getNeighboringAtoms( this.centralAtom );
        },

        getNeighboringAtoms: function ( group ) {
            return filter( this.getRadialGroups(), function ( group ) {return !group.isLonePair;} );
        },

        getLonePairNeighbors: function ( group ) {
            return filter( this.getRadialGroups(), function ( group ) {return group.isLonePair;} );
        },

        getRadialLonePairs: function () {
            return this.getLonePairNeighbors( this.centralAtom );
        },

        getGeometryConfiguration: function ( group ) {
            return model.GeometryConfiguration.getConfiguration( this.getStericNumber( group ) );
        },

        getCentralVseprConfiguration: function () {
            return this.getVseprConfiguration( this.centralAtom );
        },

        getVseprConfiguration: function ( group ) {
            return new model.VseprConfiguration( this.getNeighboringAtoms( group ).length, this.getLonePairNeighbors( group ).length );
        },

        // get the bond to the more central "parent", or undefined
        getParentBond: function ( group ) {
            // assumes we have simple atoms (star-shaped) with terminal lone pairs
            if ( group.isLonePair ) {
                return this.getBonds( group )[0];
            }
            else {
                var centralAtom = this.centralAtom;
                return phet.util.firstOrNull( this.getBonds( group ), function ( bond ) { return bond.getOtherAtom( group ) == centralAtom; } );
            }
        },

        // get the more central "parent" group
        getParent: function ( group ) {
            return this.getParentBond( group ).getOtherAtom( group );
        },

        // add in the central atom
        addCentralAtom: function ( group ) {
            this.centralAtom = group;
            this.addGroupOnly( group, true );
        },

        addGroupAndBond: function ( group, parent, bondOrder, bondLength ) {
            // add the group, but delay notifications (inconsistent state)
            this.addGroupOnly( group, false );

            bondLength = bondLength || group.position.get().minus( parent.position.get() ).magnitude() / model.PairGroup.REAL_TMP_SCALE;
            this.addBondBetween( group, parent, bondOrder, bondLength );

            // notify after bond added, so we don't send notifications in an inconsistent state
            this.onGroupAdded.updateListeners( group );
        },

        addGroupOnly: function ( group, notify ) {
            // always add the central group first
            phet.assert( this.centralAtom != null );

            this.groups.push( group );

            // notify
            if ( notify ) {
                this.onGroupAdded.updateListeners( group );
            }
        },

        addBond: function ( bond ) {
            this.bonds.push( bond );

            this.onBondAdded.updateListeners( bond );
        },

        addBondBetween: function ( a, b, order, bondLength ) {
            this.addBond( new model.Bond( a, b, order, bondLength ) );
        },

        removeBond: function ( bond ) {
            this.bonds.splice( this.bonds.indexOf( bond ), 1 );
            this.onBondRemoved.updateListeners( bond );
        },

        getCentralAtom: function () {
            return this.centralAtom;
        },

        removeGroup: function ( group ) {
            var i;

            phet.assert( this.centralAtom != group );

            // remove all of its bonds first
            var bondList = this.getBonds( group );
            for ( i = 0; i < bondList.length; i++ ) {
                phet.util.remove( this.bonds, bondList[i] );
            }

            phet.util.remove( this.groups, group );

            // notify
            this.onGroupRemoved.updateListeners( group );
            for ( i = 0; i < bondList.length; i++ ) {
                // delayed notification for bond removal
                this.onBondRemoved.updateListeners( bondList[i] );
            }
        },

        removeAllGroups: function () {
            var groupsCopy = this.groups.slice();
            for ( var i = 0; i < groupsCopy; i++ ) {
                if ( groupsCopy[i] != this.centralAtom ) {
                    this.removeGroup( groupsCopy[i] );
                }
            }
        },

        getGroups: function () {
            return this.groups;
        },

        getCorrespondingIdealGeometryVectors: function () {
            return new model.VseprConfiguration( this.getRadialAtoms().length, this.getRadialLonePairs().length ).geometry.unitVectors;
        },

        /**
         * @param bondOrder Bond order of potential pair group to add
         * @return Whether the pair group can be added, or whether this molecule would go over its pair limit
         */
        wouldAllowBondOrder: function ( bondOrder ) {
            return this.getStericNumber( this.centralAtom ) < MAX_PAIRS;
        },

        getDistantLonePairs: function () {
            return phet.util.subtract( this.getAllLonePairs(), this.getLonePairNeighbors( this.centralAtom ) );
        },

        getLocalVSEPRShape: function ( atom ) {
            var groups = model.LocalShape.sortedLonePairsFirst( this.getNeighbors( atom ) );
            var numLonePairs = phet.util.count( groups, function ( group ) {return group.isLonePair;} );
            var numAtoms = groups.length - numLonePairs;
            return new model.LocalShape( model.LocalShape.vseprPermutations( groups ), atom, groups, (new model.VseprConfiguration( numAtoms, numLonePairs )).geometry.unitVectors );
        },

        getRadialGroups: function () {
            return this.getNeighbors( this.centralAtom );
        },

        getIdealDistanceFromCenter: function ( group ) {
            // this only works on pair groups adjacent to the central atom
            var bond = this.getParentBond( group );
            phet.assert( bond.contains( this.centralAtom ) );

            return group.isLonePair ? model.PairGroup.LONE_PAIR_DISTANCE : bond.length * model.PairGroup.REAL_TMP_SCALE;
        },

        addTerminalLonePairs: function ( atom, quantity ) {
            var pairConfig = new model.VseprConfiguration( 1, quantity );
            var lonePairOrientations = pairConfig.geometry.unitVectors;
            var mapping = model.AttractorModel.findClosestMatchingConfiguration(
                    // last vector should be lowest energy (best bond if ambiguous), and is negated for the proper coordinate frame
                    [ atom.position.get().normalized() ], // TODO: why did this have to get changed to non-negated?
                    [ lonePairOrientations.get( lonePairOrientations.size() - 1 ).negated() ],
                    [phet.math.Permutation.identity( 1 ) ]
            );

            for ( var i = 0; i < quantity; i++ ) {
                // mapped into our coordinates
                var lonePairOrientation = mapping.rotateVector( lonePairOrientations[i] );
                this.addGroupAndBond( new model.PairGroup( atom.position.get().plus( lonePairOrientation.times( model.PairGroup.LONE_PAIR_DISTANCE ) ), true, false ), atom, 0 );
            }
        }
    };
})();