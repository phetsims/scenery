// Copyright 2002-2012, University of Colorado

/**
 * Arbitrary-dimensional matrix, based on Jama
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {
    phet.math.Matrix = function ( m, n, filler, fast ) {
        this.m = m;
        this.n = n;

        var size = m * n;
        this.size = size;
        var i;

        if ( fast ) {
            this.entries = filler;
        }
        else {
            if ( !filler ) {
                filler = 0;
            }

            // entries stored in row-major format
            this.entries = new Float32Array( size );

            if ( phet.util.isArray( filler ) ) {
                phet.assert( filler.length == size );

                for ( i = 0; i < size; i++ ) {
                    this.entries[i] = filler[i];
                }
            }
            else {
                for ( i = 0; i < size; i++ ) {
                    this.entries[i] = filler;
                }
            }
        }
    };

    var Matrix = phet.math.Matrix;

    /** sqrt(a^2 + b^2) without under/overflow. **/
    Matrix.hypot = function hypot( a, b ) {
        var r;
        if ( Math.abs( a ) > Math.abs( b ) ) {
            r = b / a;
            r = Math.abs( a ) * Math.sqrt( 1 + r * r );
        }
        else if ( b != 0 ) {
            r = a / b;
            r = Math.abs( b ) * Math.sqrt( 1 + r * r );
        }
        else {
            r = 0.0;
        }
        return r;
    };

    Matrix.prototype = {
        constructor: Matrix,

        copy: function () {
            var result = new Matrix( this.m, this.n );
            for ( var i = 0; i < this.size; i++ ) {
                result.entries[i] = this.entries[i];
            }
            return result;
        },

        getArray: function () {
            return this.entries;
        },

        getArrayCopy: function () {
            return new Float32Array( this.entries );
        },

        getRowDimension: function () {
            return this.m;
        },

        getColumnDimension: function () {
            return this.n;
        },

        // TODO: inline this places if we aren't using an inlining compiler! (check performance)
        index: function ( i, j ) {
            return i * this.n + j;
        },

        get: function ( i, j ) {
            return this.entries[this.index( i, j )];
        },

        set: function ( i, j, s ) {
            this.entries[this.index( i, j )] = s;
        },

        getMatrix: function ( i0, i1, j0, j1 ) {
            var result = new Matrix( i1 - i0 + 1, j1 - j0 + 1 );
            for ( var i = i0; i <= i1; i++ ) {
                for ( var j = j0; j <= j1; j++ ) {
                    result.entries[result.index( i - i0, j - j0 )] = this.entries[this.index( i, j )];
                }
            }
            return result;
        },

        // getMatrix (int[] r, int j0, int j1)
        getArrayRowMatrix: function ( r, j0, j1 ) {
            var result = new Matrix( r.length, j1 - j0 + 1 );
            for ( var i = 0; i < r.length; i++ ) {
                for ( var j = j0; j <= j1; j++ ) {
                    result.entries[result.index( i, j - j0 )] = this.entries[this.index( r[i], j )];
                }
            }
            return result;
        },

        transpose: function () {
            var result = new Matrix( this.n, this.m );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    result.entries[result.index( j, i )] = this.entries[this.index( i, j )];
                }
            }
            return result;
        },

        norm1: function () {
            var f = 0;
            for ( var j = 0; j < this.n; j++ ) {
                var s = 0;
                for ( var i = 0; i < this.m; i++ ) {
                    s += Math.abs( this.entries[ this.index( i, j ) ] );
                }
                f = Math.max( f, s );
            }
            return f;
        },

        norm2: function () {
            return (new phet.math.SingularValueDecomposition( this ).norm2());
        },

        normInf: function () {
            var f = 0;
            for ( var i = 0; i < this.m; i++ ) {
                var s = 0;
                for ( var j = 0; j < this.n; j++ ) {
                    s += Math.abs( this.entries[ this.index( i, j ) ] );
                }
                f = Math.max( f, s );
            }
            return f;
        },

        normF: function () {
            var f = 0;
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    f = hypot( f, this.entries[ this.index( i, j ) ] );
                }
            }
            return f;
        },

        uminus: function () {
            var result = new Matrix( this.m, this.n );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    result.entries[result.index( i, j )] = -this.entries[ this.index( i, j ) ];
                }
            }
            return result;
        },

        plus: function ( matrix ) {
            this.checkMatrixDimensions( matrix );
            var result = new Matrix( this.m, this.n );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = result.index( i, j );
                    result.entries[index] = this.entries[index] + matrix.entries[index];
                }
            }
            return result;
        },

        plusEquals: function ( matrix ) {
            this.checkMatrixDimensions( matrix );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = result.index( i, j );
                    this.entries[index] = this.entries[index] + matrix.entries[index];
                }
            }
            return this;
        },

        minus: function ( matrix ) {
            this.checkMatrixDimensions( matrix );
            var result = new Matrix( this.m, this.n );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = this.index( i, j );
                    result.entries[index] = this.entries[index] - matrix.entries[index];
                }
            }
            return result;
        },

        minusEquals: function ( matrix ) {
            this.checkMatrixDimensions( matrix );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = this.index( i, j );
                    this.entries[index] = this.entries[index] - matrix.entries[index];
                }
            }
            return this;
        },

        arrayTimes: function ( matrix ) {
            this.checkMatrixDimensions( matrix );
            var result = new Matrix( this.m, this.n );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = result.index( i, j );
                    result.entries[index] = this.entries[index] * matrix.entries[index];
                }
            }
            return result;
        },

        arrayTimesEquals: function ( matrix ) {
            this.checkMatrixDimensions( matrix );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = this.index( i, j );
                    this.entries[index] = this.entries[index] * matrix.entries[index];
                }
            }
            return this;
        },

        arrayRightDivide: function ( matrix ) {
            this.checkMatrixDimensions( matrix );
            var result = new Matrix( this.m, this.n );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = this.index( i, j );
                    result.entries[index] = this.entries[index] / matrix.entries[index];
                }
            }
            return result;
        },

        arrayRightDivideEquals: function ( matrix ) {
            this.checkMatrixDimensions( matrix );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = this.index( i, j );
                    this.entries[index] = this.entries[index] / matrix.entries[index];
                }
            }
            return this;
        },

        arrayLeftDivide: function ( matrix ) {
            this.checkMatrixDimensions( matrix );
            var result = new Matrix( this.m, this.n );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = this.index( i, j );
                    result.entries[index] = matrix.entries[index] / this.entries[index];
                }
            }
            return result;
        },

        arrayLeftDivideEquals: function ( matrix ) {
            this.checkMatrixDimensions( matrix );
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = this.index( i, j );
                    this.entries[index] = matrix.entries[index] / this.entries[index];
                }
            }
            return this;
        },

        times: function ( matrixOrScalar ) {
            var result;
            var i, j, k, s;
            var matrix;
            if ( matrixOrScalar.isMatrix ) {
                matrix = matrixOrScalar;
                if ( matrix.m != this.n ) {
                    throw new Error( "Matrix inner dimensions must agree." );
                }
                result = new Matrix( this.m, matrix.n );
                var matrixcolj = new Float32Array( this.n );
                for ( j = 0; j < matrix.n; j++ ) {
                    for ( k = 0; k < this.n; k++ ) {
                        matrixcolj[k] = matrix.entries[ matrix.index( k, j ) ];
                    }
                    for ( i = 0; i < this.m; i++ ) {
                        s = 0;
                        for ( k = 0; k < this.n; k++ ) {
                            s += this.entries[this.index( i, k )] * matrixcolj[k];
                        }
                        result.entries[result.index( i, j )] = s;
                    }
                }
                return result;
            }
            else {
                s = matrixOrScalar;
                result = new Matrix( this.m, this.n );
                for ( i = 0; i < this.m; i++ ) {
                    for ( j = 0; j < this.n; j++ ) {
                        result.entries[result.index( i, j )] = s * this.entries[this.index( i, j )];
                    }
                }
                return result;
            }
        },

        timesEquals: function ( s ) {
            for ( var i = 0; i < this.m; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    var index = this.index( i, j );
                    this.entries[index] = s * this.entries[index];
                }
            }
            return this;
        },

        solve: function ( matrix ) {
            return (this.m == this.n ? (new phet.math.LUDecomposition( this )).solve( matrix ) :
                    (new phet.math.QRDecomposition( this )).solve( matrix ));
        },

        solveTranspose: function ( matrix ) {
            return this.transpose().solve( matrix.transpose() );
        },

        inverse: function () {
            return this.solve( Matrix.identity( this.m, this.m ) );
        },

        det: function () {
            return new phet.math.LUDecomposition( this ).det();
        },

        rank: function () {
            return new phet.math.SingularValueDecomposition( this ).rank();
        },

        cond: function () {
            return new phet.math.SingularValueDecomposition( this ).cond();
        },

        trace: function () {
            var t = 0;
            for ( var i = 0; i < Math.min( this.m, this.n ); i++ ) {
                t += this.entries[ this.index( i, i ) ];
            }
            return t;
        },

        checkMatrixDimensions: function ( matrix ) {
            if ( matrix.m != this.m || matrix.n != this.n ) {
                throw new Error( "Matrix dimensions must agree." );
            }
        },

        toString: function () {
            var result = "";
            result += "dim: " + this.getRowDimension() + "x" + this.getColumnDimension() + "\n";
            for ( var row = 0; row < this.getRowDimension(); row++ ) {
                for ( var col = 0; col < this.getColumnDimension(); col++ ) {
                    result += this.get( row, col ) + " ";
                }
                result += "\n";
            }
            return result;
        },

        // returns a vector that is contained in the specified column
        extractVector2: function ( column ) {
            phet.assert( this.m == 2 ); // rows should match vector dimension
            return new phet.math.Vector2( this.get( 0, column ), this.get( 1, column ) );
        },

        // returns a vector that is contained in the specified column
        extractVector3: function ( column ) {
            phet.assert( this.m == 3 ); // rows should match vector dimension
            return new phet.math.Vector3( this.get( 0, column ), this.get( 1, column ), this.get( 2, column ) );
        },

        // returns a vector that is contained in the specified column
        extractVector4: function ( column ) {
            phet.assert( this.m == 4 ); // rows should match vector dimension
            return new phet.math.Vector4( this.get( 0, column ), this.get( 1, column ), this.get( 2, column ), this.get( 3, column ) );
        },

        isMatrix: true
    };

    Matrix.identity = function ( m, n ) {
        var result = new Matrix( m, n );
        for ( var i = 0; i < m; i++ ) {
            for ( var j = 0; j < n; j++ ) {
                result.entries[result.index( i, j )] = (i == j ? 1.0 : 0.0);
            }
        }
        return result;
    };

    Matrix.rowVector2 = function ( vector ) {
        return new Matrix( 1, 2, [vector.x, vector.y] );
    };

    Matrix.rowVector3 = function ( vector ) {
        return new Matrix( 1, 3, [vector.x, vector.y, vector.z] );
    };

    Matrix.rowVector4 = function ( vector ) {
        return new Matrix( 1, 4, [vector.x, vector.y, vector.z, vector.w] );
    };

    Matrix.rowVector = function ( vector ) {
        if ( vector.isVector2 ) {
            return Matrix.rowVector2( vector );
        }
        else if ( vector.isVector3 ) {
            return Matrix.rowVector3( vector );
        }
        else if ( vector.isVector4 ) {
            return Matrix.rowVector4( vector );
        }
        else {
            throw new Error( "undetected type of vector: " + vector.toString() );
        }
    };

    Matrix.columnVector2 = function ( vector ) {
        return new Matrix( 2, 1, [vector.x, vector.y] );
    };

    Matrix.columnVector3 = function ( vector ) {
        return new Matrix( 3, 1, [vector.x, vector.y, vector.z] );
    };

    Matrix.columnVector4 = function ( vector ) {
        return new Matrix( 4, 1, [vector.x, vector.y, vector.z, vector.w] );
    };

    Matrix.columnVector = function ( vector ) {
        if ( vector.isVector2 ) {
            return Matrix.columnVector2( vector );
        }
        else if ( vector.isVector3 ) {
            return Matrix.columnVector3( vector );
        }
        else if ( vector.isVector4 ) {
            return Matrix.columnVector4( vector );
        }
        else {
            throw new Error( "undetected type of vector: " + vector.toString() );
        }
    };

    /**
     * Create a Matrix where each column is a vector
     */

    Matrix.fromVectors2 = function ( vectors ) {
        var dimension = 2;
        var n = vectors.length;
        var data = new Float32Array( dimension * n );

        for ( var i = 0; i < n; i++ ) {
            var vector = vectors[i];
            data[i] = vector.x;
            data[i + n] = vector.y;
        }

        return new Matrix( dimension, n, data, true );
    };

    Matrix.fromVectors3 = function ( vectors ) {
        var dimension = 3;
        var n = vectors.length;
        var data = new Float32Array( dimension * n );

        for ( var i = 0; i < n; i++ ) {
            var vector = vectors[i];
            data[i] = vector.x;
            data[i + n] = vector.y;
            data[i + 2 * n] = vector.z;
        }

        return new Matrix( dimension, n, data, true );
    };

    Matrix.fromVectors4 = function ( vectors ) {
        var dimension = 4;
        var n = vectors.length;
        var data = new Float32Array( dimension * n );

        for ( var i = 0; i < n; i++ ) {
            var vector = vectors[i];
            data[i] = vector.x;
            data[i + n] = vector.y;
            data[i + 2 * n] = vector.z;
            data[i + 3 * n] = vector.w;
        }

        return new Matrix( dimension, n, data, true );
    };

})();
