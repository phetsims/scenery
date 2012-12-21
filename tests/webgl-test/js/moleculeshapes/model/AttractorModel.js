// Copyright 2002-2012, University of Colorado

/**
 * Contains the logic for applying an "attractor" force to a molecule that first:
 * (1) finds the closest VSEPR configuration (with rotation) to our current positions, and
 * (2) pushes the electron pairs towards those positions.
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.model = phet.moleculeshapes.model || {};

// create a new scope
(function () {

    var PairGroup = phet.moleculeshapes.model.PairGroup;

    // just static calls, so just create an empty object
    phet.moleculeshapes.model.AttractorModel = { };

    var AttractorModel = phet.moleculeshapes.model.AttractorModel;

    /**
     * Apply an attraction to the closest ideal position, with the given time elapsed
     *
     * @param groups                An ordered list of pair groups that should be considered, along with the relevant permutations
     * @param timeElapsed           Time elapsed
     * @param idealOrientations     An ideal position, that may be rotated.
     * @param allowablePermutations The un-rotated stable position that we are attracted towards
     * @param center                The point that the groups should be rotated around. Usually a central atom that all of the groups connect to
     * @return A measure of total error (least squares-style)
     */
    AttractorModel.applyAttractorForces = function ( groups, timeElapsed, idealOrientations, allowablePermutations, center, angleRepulsion, lastPermutation ) {
        var currentOrientations = phet.util.map( groups, function ( group ) {
            return group.position.get().minus( center ).normalized();
        } );
        var mapping = AttractorModel.findClosestMatchingConfiguration( currentOrientations, idealOrientations, allowablePermutations, lastPermutation );

        var aroundCenterAtom = center.equals( new phet.math.Vector3() );

        var totalDeltaMagnitude = 0;
        var i;

        // for each electron pair, push it towards its computed target
        for ( i = 0; i < groups.length; i++ ) {

            var pair = groups[i];

            var targetOrientation = mapping.target.extractVector3( i );
            var currentMagnitude = ( pair.position.get().minus( center ) ).magnitude();
            var targetLocation = targetOrientation.times( currentMagnitude ).plus( center );

            var delta = targetLocation.minus( pair.position.get() );
            totalDeltaMagnitude += delta.magnitude() * delta.magnitude();

            /*
             * NOTE: adding delta here effectively is squaring the distance, thus more force when far from the target,
             * and less force when close to the target. This is important, since we want more force in a potentially
             * otherwise-stable position, and less force where our coulomb-like repulsion will settle it into a stable
             * position
             */
            var strength = timeElapsed * 3 * delta.magnitude();

            // change the velocity of all of the pairs, unless it is an atom at the origin!
            if ( pair.isLonePair || !pair.isCentralAtom() ) {
                if ( aroundCenterAtom ) {
                    pair.addVelocity( delta.times( strength ) );
                }
            }

            // position movement for faster convergence
            if ( !pair.isCentralAtom() && aroundCenterAtom ) { // TODO: better way of not moving the center atom?
                pair.position.set( pair.position.get().plus( delta.times( 2.0 * timeElapsed ) ) );
            }

            // if we are a terminal lone pair, move us just with this but much more quickly
            if ( !pair.isCentralAtom() && !aroundCenterAtom ) {
//                pair.position.set( targetLocation );
                pair.position.set( pair.position.get().plus( delta.times( Math.min( 20.0 * timeElapsed, 1 ) ) ) );
            }
        }

        var error = Math.sqrt( totalDeltaMagnitude );

        // angle-based repulsion
        if ( angleRepulsion && aroundCenterAtom ) {
            var pairIndexList = phet.util.pairs( phet.util.rangeInclusive( 0, groups.length - 1 ) );
            for ( i = 0; i < pairIndexList.length; i++ ) {
                var pairIndices = pairIndexList[i];
                var aIndex = pairIndices[0];
                var bIndex = pairIndices[1];
                var a = groups[ aIndex ];
                var b = groups[bIndex ];

                // current orientations w.r.t. the center
                var aOrientation = a.position.get().minus( center ).normalized();
                var bOrientation = b.position.get().minus( center ).normalized();

                // desired orientations
                var aTarget = mapping.target.extractVector3( aIndex ).normalized();
                var bTarget = mapping.target.extractVector3( bIndex ).normalized();
                var targetAngle = Math.acos( phet.math.clamp( aTarget.dot( bTarget ), -1, 1 ) );
                var currentAngle = Math.acos( phet.math.clamp( aOrientation.dot( bOrientation ), -1, 1 ) );
                var angleDifference = ( targetAngle - currentAngle );

                var dirTowardsA = a.position.get().minus( b.position.get() ).normalized();
                var timeFactor = PairGroup.getTimescaleImpulseFactor( timeElapsed );

                var extraClosePushFactor = phet.math.clamp( 3 * Math.pow( Math.PI - currentAngle, 2 ) / ( Math.PI * Math.PI ), 1, 3 );

                var push = dirTowardsA.times( timeFactor
                                                      * angleDifference
                                                      * PairGroup.ANGLE_REPULSION_SCALE
                                                      * ( currentAngle < targetAngle ? 2.0 : 0.5 )
                                                      * extraClosePushFactor );
                a.addVelocity( push );
                b.addVelocity( push.negated() );
            }
        }

        return error;
    };

    /**
     * Find the closest VSEPR configuration for a particular molecule. Conceptually, we iterate through
     * each possible valid 1-to-1 mapping from electron pair to direction in our VSEPR geometry. For each
     * mapping, we calculate the rotation that makes the best match, and then calculate the error. We return
     * a result for the mapping (permutation) with the lowest error.
     * <p/>
     * This uses a slightly modified rotation computation from http://igl.ethz.ch/projects/ARAP/svd_rot.pdf
     * (Least-Squares Rigid Motion Using SVD). Basically, we ignore the centroid and translation computations,
     * since we want everything to be rotated around the origin. We also don't weight the individual electron
     * pairs.
     * <p/>
     * Of note, the lower-index slots in the VseprConfiguration (GeometryConfiguration) are for higher-repulsion
     * pair groups (the order is triple > double > lone pair > single). We need to iterate through all permutations,
     * but with the repulsion-ordering constraint (no single bond will be assigned a lower-index slot than a lone pair)
     * so we end up splitting the potential slots into bins for each repulsion type and iterating over all of the permutations.
     *
     * @param currentOrientations   An ordered list of orientations (normalized) that should be considered, along with the relevant permutations
     * @param idealOrientations     The un-rotated stable position that we are attracted towards
     * @param allowablePermutations A list of permutations that map stable positions to pair groups in order.
     * @return Result mapping (see docs there)
     */
    AttractorModel.findClosestMatchingConfiguration = function ( currentOrientations, idealOrientations, allowablePermutations, lastPermutation ) {
        var n = currentOrientations.length; // number of total pairs

        // y == electron pair positions
        var y = phet.math.Matrix.fromVectors3( currentOrientations );
        var yTransposed = y.transpose();

        // closure over constant variables
        function calculateTarget( permutation ) {
            // x == configuration positions
            var x = phet.math.Matrix.fromVectors3( permutation.apply( idealOrientations ) );

            // compute the rotation matrix
            var rot = AttractorModel.computeRotationMatrixWithTranspose( x, yTransposed );

            // target matrix, same shape as our y (current position) matrix
            var target = rot.times( x );

            // calculate the error
            var error = 0;
            var offsets = y.minus( target );
            var squaredOffsets = offsets.arrayTimes( offsets );
            for ( var i = 0; i < n; i++ ) {
                error += squaredOffsets.get( 0, i ) + squaredOffsets.get( 1, i ) + squaredOffsets.get( 2, i );
            }

            return new AttractorModel.ResultMapping( error, target, permutation, rot );
        }

        var bestResult = lastPermutation !== undefined ? calculateTarget( lastPermutation ) : null;

        // TODO: log how effective the permutation checking is at removing the search space
        for ( var pIndex = 0; pIndex < allowablePermutations.length; pIndex++ ) {
            var permutation = allowablePermutations[pIndex];

            if( n > 2 && bestResult != null && bestResult.permutation != permutation ) {
                var permutedOrientations = permutation.apply( idealOrientations );
                var errorLowBound = 4 - 4 * Math.cos( Math.abs(
                    Math.acos( permutedOrientations[0].dot( currentOrientations[0] ) )
                    - Math.acos( permutedOrientations[1].dot( currentOrientations[1] ) )
                 ) );

                // throw out results where this arbitrarily-chosen lower bound rules out the entire permutation
                if( bestResult.error < errorLowBound ) {
                    continue;
                }
            }

            var result = calculateTarget( permutation );

            if ( bestResult == null || result.error < bestResult.error ) {
                bestResult = result;
            }
        }
        return bestResult;
    };

    AttractorModel.getOrientationsFromOrigin = function ( groups ) {
        return phet.util.map( groups, function ( group ) {
            return group.position.get().normalized();
        } );
    };

    AttractorModel.computeRotationMatrixWithTranspose = function ( x, yTransposed ) {
        // S = X * Y^T
        var s = x.times( yTransposed );

        // this code will loop infinitely on NaN, so we want to double-check
        phet.assert( !isNaN( s.get( 0, 0 ) ) );
        var svd = new phet.math.SingularValueDecomposition( s );
        var det = svd.getV().times( svd.getU().transpose() ).det();

        return svd.getV().times( new phet.math.Matrix( 3, 3, [1, 0, 0, 0, 1, 0, 0, 0, det] ).times( svd.getU().transpose() ) );
    };

    // double error, Matrix target, Permutation permutation, Matrix rotation
    AttractorModel.ResultMapping = function ( error, target, permutation, rotation ) {
        this.error = error;
        this.target = target;
        this.permutation = permutation;
        this.rotation = rotation;
    };

    AttractorModel.ResultMapping.prototype = {
        constructor: AttractorModel.ResultMapping,

        rotateVector: function ( v ) {
            var x = phet.math.Matrix.columnVector3( v );
            var rotated = this.rotation.times( x );
            return rotated.extractVector3( 0 );
        }
    };

    /**
     * Call the function with each individual permutation of the list elements of "lists"
     *
     * @param lists    List of lists. Order of lists will not change, however each possible permutation involving sub-lists will be used
     * @param callback Function to call
     */
    AttractorModel.forEachMultiplePermutations = function ( lists, callback ) {
        if ( lists.length == 0 ) {
            callback.call( undefined, lists );
        }
        else {
            // make a copy of 'lists'
            var remainder = lists.slice( 0 );
            var first = remainder[0];

            remainder.splice( 0, 1 );

            AttractorModel.forEachPermutation( first, [], function ( permutedFirst ) {
                AttractorModel.forEachMultiplePermutations( remainder, function ( subLists ) {
                    var arr = new Array( lists.length );
                    arr[0] = permutedFirst;
                    for ( var i = 0; i < subLists.length; i++ ) {
                        arr[i + 1] = subLists[i];
                    }
                    callback.call( undefined, arr );
                } );
            } );
        }
    };

    /**
     * Call our function with each permutation of the provided list PREFIXED by prefix, in lexicographic order
     *
     * @param list     List to generate permutations of
     * @param prefix   Elements that should be inserted at the front of each list before each call
     * @param callback Function to call
     */
    AttractorModel.forEachPermutation = function ( list, prefix, callback ) {
        if ( list.length == 0 ) {
            callback.call( undefined, prefix );
        }
        else {
            for ( var i = 0; i < list.length; i++ ) {
                var element = list[i];

                var newList = list.slice();
                newList.splice( newList.indexOf( element ), 1 );

                var newPrefix = prefix.slice();
                newPrefix.push( element );

                AttractorModel.forEachPermutation( newList, newPrefix, callback );
            }
        }
    };

    AttractorModel.listPrint = function ( lists ) {
        var ret = "";
        for ( var i = 0; i < lists.length; i++ ) {
            var list = lists[i];
            ret += " ";
            for ( var j = 0; j < list.length; j++ ) {
                ret += list[j].toString();
            }
        }
        return ret;
    };

    AttractorModel.testMe = function () {
        /*
         Testing of permuting each individual list. Output:
         AB C DEF
         AB C DFE
         AB C EDF
         AB C EFD
         AB C FDE
         AB C FED
         BA C DEF
         BA C DFE
         BA C EDF
         BA C EFD
         BA C FDE
         BA C FED
         */

        var arr = [
            ["A", "B"],
            ["C"],
            ["D", "E", "F"]
        ];

        AttractorModel.forEachMultiplePermutations( arr, function ( lists ) {
            console.log( AttractorModel.listPrint( lists ) );
        } );
    };
})();
