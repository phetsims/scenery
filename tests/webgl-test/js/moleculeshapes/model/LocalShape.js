// Copyright 2002-2012, University of Colorado

/**
 * The ideal local shape for a certain central atom and its (local) neighbors.
 * <p/>
 * Also contains the ability to push the local atoms into place, along with many helper functions
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.model = phet.moleculeshapes.model || {};

// create a new scope
(function () {
    var model = phet.moleculeshapes.model;

    phet.moleculeshapes.model.LocalShape = function ( allowedPermutations, centralAtom, groups, idealOrientations ) {
        // denotes how we can map the groups into the orientation vectors. some combinations may not be possible
        this.allowedPermutations = allowedPermutations;

        // all of our pair groups should be connected to this atom
        this.centralAtom = centralAtom;


        this.groups = groups;

        // the ideal orientations (unit vectors) for the groups representing the ideal local shape
        this.idealOrientations = idealOrientations;
    };

    var LocalShape = phet.moleculeshapes.model.LocalShape;

    /**
     * Given a list of permutations, return all permutations that exist with the specified indices permuted in all different ways.
     * <p/>
     * IE, if given the list of the single permutation (12), and specified indices {3,4,5}, the permutations returned will be
     * (12)(34),(12)(35),(12)(45),(12)(453),(12)(534),(12)
     */
    LocalShape.permuteListWithIndices = function ( permutations, indices ) {
        if ( indices.length < 2 ) {
            // no changes if we can't move more than 1 element (need somewhere to put it)
            return permutations;
        }
        var result = [];
        for ( var i = 0; i < permutations.length; i++ ) {
            var permutation = permutations[i];

            var resultsToAdd = permutation.withIndicesPermuted( indices );
            for ( var j = 0; j < resultsToAdd.length; j++ ) {
                result.push( resultsToAdd[j] );
            }
        }
        return result;
    };

    LocalShape.sortedLonePairsFirst = function ( groups ) {
        var result = groups.slice();
        result.sort( function ( a, b ) {
            if ( a.isLonePair == b.isLonePair ) {
                return 0;
            }
            else if ( a.isLonePair ) {
                return -1;
            }
            else {
                return 1;
            }
        } );
        return result;
    };

    // allow switching of lone pairs with each other, and all other types of bonds with each other
    // NOTE: I recommended double or triple bonds being put in "higher repulsion" spots over single bonds, but this was specifically rejected. -JO
    LocalShape.vseprPermutations = function ( neighbors ) {
        var permutations = [];
        permutations.push( phet.math.Permutation.identity( neighbors.length ) );

        var indexOf = function ( group ) {
            return neighbors.indexOf( group );
        };

        // partition the neighbors into lone pairs and atoms.
        var partitioned = phet.util.partition( neighbors, function ( group ) {
            return group.isLonePair;
        } );
        // this separation looks better in languages where you say "(lonePairs, atoms) = partition(...)"
        var lonePairs = partitioned[0];
        var atoms = partitioned[1];

        // permute away the lone pairs
        permutations = LocalShape.permuteListWithIndices( permutations, phet.util.map( lonePairs, indexOf ) );

        // permute away the bonded groups
        permutations = LocalShape.permuteListWithIndices( permutations, phet.util.map( atoms, indexOf ) );
        return permutations;
    };

    // allow switching of lone pairs with each other, and all other types of bonds with the same type of element
    LocalShape.realPermutations = function ( neighbors ) {
        var permutations = [];
        permutations.add( phet.math.Permutation.identity( neighbors.length ) );

        var indexOf = function ( group ) {
            return neighbors.indexOf( group );
        };

        // allow interchanging of lone pairs
        var lonePairs = phet.util.filter( neighbors, function ( group ) { return group.isLonePair} );
        permutations = LocalShape.permuteListWithIndices( permutations, phet.util.map( lonePairs, indexOf ) );

        // allow interchanging of pair groups when they have the same chemical element
        var atoms = phet.util.filter( neighbors, function ( group ) { return !group.isLonePair} );

        var usedElements = phet.util.unique( phet.util.map( atoms, function ( group ) { return group.getElement();} ) );

        for ( var i = 0; i < usedElements.length; i++ ) {
            var element = usedElements[i];

            // since the closure is being executed at this point, the warning in this line can be ignored
            var atomsWithElement = phet.util.filter( atoms, function ( group ) {return group.getElement() == element; } );
            permutations = LocalShape.permuteListWithIndices( permutations, phet.util.map( atomsWithElement, indexOf ) );
        }

        return permutations;
    };

    LocalShape.prototype = {
        constructor: LocalShape,

        /**
         * Attracts the atoms to their ideal shape, and returns the current approximate "error" that they have at this state.
         * <p/>
         * Attraction done by adding in velocity.
         *
         * @param tpf Time elapsed.
         * @return Amount of error (least squares-style)
         */
        applyAttraction: function ( tpf ) {
            return model.AttractorModel.applyAttractorForces( this.groups, tpf, this.idealOrientations, this.allowedPermutations, this.centralAtom.position.get(), false );
        },

        applyAngleAttractionRepulsion: function ( tpf ) {
            model.AttractorModel.applyAttractorForces( this.groups, tpf, this.idealOrientations, this.allowedPermutations, this.centralAtom.position.get(), true );
        }
    };
})();
