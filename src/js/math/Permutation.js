// Copyright 2002-2012, University of Colorado

/**
 * An immutable permutation that can permute an array
 *
 * @author Jonathan Olson
 */

// ensure proper namespace
var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {

    // Creates a permutation that will rearrange a list so that newList[i] = oldList[permutation[i]]
    phet.math.Permutation = function ( indices ) {
        this.indices = indices;
    };

    // shortcut within the scope
    var Permutation = phet.math.Permutation;

    // An identity permutation with a specific number of elements
    Permutation.identity = function ( size ) {
        phet.assert( size >= 0 );
        var indices = new Array( size );
        for ( var i = 0; i < size; i++ ) {
            indices[i] = i;
        }
        return new Permutation( indices );
    };

    // lists all permutations that have a given size
    Permutation.permutations = function ( size ) {
        var result = new Array();
        Permutation.forEachPermutation( phet.util.rangeInclusive( 0, size - 1 ), function ( integers ) {
            result.push( new Permutation( integers ) );
        } );
        return result;
    };

    /**
     * Call our function with each permutation of the provided list PREFIXED by prefix, in lexicographic order
     *
     * @param array     List to generate permutations of
     * @param prefix   Elements that should be inserted at the front of each list before each call
     * @param callback Function to call
     */
    function recursiveForEachPermutation( array, prefix, callback ) {
        if ( array.length == 0 ) {
            callback.call( undefined, prefix );
        }
        else {
            for ( var i = 0; i < array.length; i++ ) {
                var element = array[i];

                // remove the element from the array
                var nextArray = array.slice( 0 );
                nextArray.splice( i, 1 );

                // add it into the prefix
                var nextPrefix = prefix.slice( 0 );
                nextPrefix.push( element );

                recursiveForEachPermutation( nextArray, nextPrefix, callback );
            }
        }
    }

    Permutation.forEachPermutation = function ( array, callback ) {
        recursiveForEachPermutation( array, new Array(), callback );
    };

    phet.math.Permutation.prototype = {
        constructor: Permutation,

        size: function () {
            return this.indices.length;
        },

        apply: function ( arrayOrInt ) {
            if ( phet.util.isArray( arrayOrInt ) ) {
                if ( arrayOrInt.length != this.size() ) {
                    throw new Error( "Permutation length " + this.size() + " not equal to list length " + arrayOrInt.length )
                }

                // permute it as an array
                var result = new Array( arrayOrInt.length );
                for ( var i = 0; i < arrayOrInt.length; i++ ) {
                    result[i] = arrayOrInt[ this.indices[i] ];
                }
                return result;
            }
            else {
                // permute a single index
                return this.indices[ arrayOrInt ];
            }
        },

        // The inverse of this permutation
        inverted: function () {
            var newPermutation = new Array( this.size() );
            for ( var i = 0; i < this.size(); i++ ) {
                newPermutation[this.indices[i]] = i;
            }
            return new Permutation( newPermutation );
        },

        withIndicesPermuted: function ( indices ) {
            var result = new Array();
            var that = this;
            Permutation.forEachPermutation( indices, function ( integers ) {
                var oldIndices = that.indices;
                var newPermutation = oldIndices.slice( 0 );

                for ( var i = 0; i < indices.length; i++ ) {
                    newPermutation[indices[i]] = oldIndices[integers[i]];
                }
                result.push( new Permutation( newPermutation ) );
            } );
            return result;
        },

        toString: function () {
            return "P[" + phet.util.mkString( this.indices, ", " ) + "]";
        }
    };

    Permutation.testMe = function () {
        var a = new Permutation( [ 1, 4, 3, 2, 0 ] );
        console.log( a.toString() );

        var b = a.inverted();
        console.log( b.toString() );

        console.log( b.withIndicesPermuted( [ 0, 3, 4 ] ).toString() );

        console.log( Permutation.permutations( 4 ).toString() );
    }
})();