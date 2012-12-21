// Copyright 2002-2012, University of Colorado

/**
 * SVD decomposition, based on Jama
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {
    var Matrix = phet.math.Matrix;

    phet.math.SingularValueDecomposition = function ( matrix ) {
        this.matrix = matrix;

        var Arg = matrix;

        // Derived from LINPACK code.
        // Initialize.
        var A = Arg.getArrayCopy();
        this.m = Arg.getRowDimension();
        this.n = Arg.getColumnDimension();
        var m = this.m;
        var n = this.n;

        var min = Math.min;
        var max = Math.max;
        var pow = Math.pow;
        var abs = Math.abs;

        /* Apparently the failing cases are only a proper subset of (m<n),
         so let's not throw error.  Correct fix to come later?
         if (m<n) {
         throw new IllegalArgumentException("Jama SVD only works for m >= n"); }
         */
        var nu = min( m, n );
        this.s = new Float32Array( min( m + 1, n ) );
        var s = this.s;
        this.U = new Float32Array( m * nu );
        var U = this.U;
        this.V = new Float32Array( n * n );
        var V = this.V;
        var e = new Float32Array( n );
        var work = new Float32Array( m );
        var wantu = true;
        var wantv = true;

        var i, j, k, t, f;

        var hypot = Matrix.hypot;

        // Reduce A to bidiagonal form, storing the diagonal elements
        // in s and the super-diagonal elements in e.

        var nct = min( m - 1, n );
        var nrt = max( 0, min( n - 2, m ) );
        for ( k = 0; k < max( nct, nrt ); k++ ) {
            if ( k < nct ) {

                // Compute the transformation for the k-th column and
                // place the k-th diagonal in s[k].
                // Compute 2-norm of k-th column without under/overflow.
                s[k] = 0;
                for ( i = k; i < m; i++ ) {
                    s[k] = hypot( s[k], A[i * n + k] );
                }
                if ( s[k] != 0.0 ) {
                    if ( A[k * n + k] < 0.0 ) {
                        s[k] = -s[k];
                    }
                    for ( i = k; i < m; i++ ) {
                        A[i * n + k] /= s[k];
                    }
                    A[k * n + k] += 1.0;
                }
                s[k] = -s[k];
            }
            for ( j = k + 1; j < n; j++ ) {
                if ( (k < nct) && (s[k] != 0.0) ) {

                    // Apply the transformation.

                    t = 0;
                    for ( i = k; i < m; i++ ) {
                        t += A[i * n + k] * A[i * n + j];
                    }
                    t = -t / A[k * n + k];
                    for ( i = k; i < m; i++ ) {
                        A[i * n + j] += t * A[i * n + k];
                    }
                }

                // Place the k-th row of A into e for the
                // subsequent calculation of the row transformation.

                e[j] = A[k * n + j];
            }
            if ( wantu && (k < nct) ) {

                // Place the transformation in U for subsequent back
                // multiplication.

                for ( i = k; i < m; i++ ) {
                    U[i * nu + k] = A[i * n + k];
                }
            }
            if ( k < nrt ) {

                // Compute the k-th row transformation and place the
                // k-th super-diagonal in e[k].
                // Compute 2-norm without under/overflow.
                e[k] = 0;
                for ( i = k + 1; i < n; i++ ) {
                    e[k] = hypot( e[k], e[i] );
                }
                if ( e[k] != 0.0 ) {
                    if ( e[k + 1] < 0.0 ) {
                        e[k] = -e[k];
                    }
                    for ( i = k + 1; i < n; i++ ) {
                        e[i] /= e[k];
                    }
                    e[k + 1] += 1.0;
                }
                e[k] = -e[k];
                if ( (k + 1 < m) && (e[k] != 0.0) ) {

                    // Apply the transformation.

                    for ( i = k + 1; i < m; i++ ) {
                        work[i] = 0.0;
                    }
                    for ( j = k + 1; j < n; j++ ) {
                        for ( i = k + 1; i < m; i++ ) {
                            work[i] += e[j] * A[i * n + j];
                        }
                    }
                    for ( j = k + 1; j < n; j++ ) {
                        t = -e[j] / e[k + 1];
                        for ( i = k + 1; i < m; i++ ) {
                            A[i * n + j] += t * work[i];
                        }
                    }
                }
                if ( wantv ) {

                    // Place the transformation in V for subsequent
                    // back multiplication.

                    for ( i = k + 1; i < n; i++ ) {
                        V[i * n + k] = e[i];
                    }
                }
            }
        }

        // Set up the final bidiagonal matrix or order p.

        var p = min( n, m + 1 );
        if ( nct < n ) {
            s[nct] = A[nct * n + nct];
        }
        if ( m < p ) {
            s[p - 1] = 0.0;
        }
        if ( nrt + 1 < p ) {
            e[nrt] = A[nrt * n + p - 1];
        }
        e[p - 1] = 0.0;

        // If required, generate U.

        if ( wantu ) {
            for ( j = nct; j < nu; j++ ) {
                for ( i = 0; i < m; i++ ) {
                    U[i * nu + j] = 0.0;
                }
                U[j * nu + j] = 1.0;
            }
            for ( k = nct - 1; k >= 0; k-- ) {
                if ( s[k] != 0.0 ) {
                    for ( j = k + 1; j < nu; j++ ) {
                        t = 0;
                        for ( i = k; i < m; i++ ) {
                            t += U[i * nu + k] * U[i * nu + j];
                        }
                        t = -t / U[k * nu + k];
                        for ( i = k; i < m; i++ ) {
                            U[i * nu + j] += t * U[i * nu + k];
                        }
                    }
                    for ( i = k; i < m; i++ ) {
                        U[i * nu + k] = -U[i * nu + k];
                    }
                    U[k * nu + k] = 1.0 + U[k * nu + k];
                    for ( i = 0; i < k - 1; i++ ) {
                        U[i * nu + k] = 0.0;
                    }
                }
                else {
                    for ( i = 0; i < m; i++ ) {
                        U[i * nu + k] = 0.0;
                    }
                    U[k * nu + k] = 1.0;
                }
            }
        }

        // If required, generate V.

        if ( wantv ) {
            for ( k = n - 1; k >= 0; k-- ) {
                if ( (k < nrt) && (e[k] != 0.0) ) {
                    for ( j = k + 1; j < nu; j++ ) {
                        t = 0;
                        for ( i = k + 1; i < n; i++ ) {
                            t += V[i * n + k] * V[i * n + j];
                        }
                        t = -t / V[k + 1 * n + k];
                        for ( i = k + 1; i < n; i++ ) {
                            V[i * n + j] += t * V[i * n + k];
                        }
                    }
                }
                for ( i = 0; i < n; i++ ) {
                    V[i * n + k] = 0.0;
                }
                V[k * n + k] = 1.0;
            }
        }

        // Main iteration loop for the singular values.

        var pp = p - 1;
        var iter = 0;
        var eps = pow( 2.0, -52.0 );
        var tiny = pow( 2.0, -966.0 );
        while ( p > 0 ) {
            var kase;

            // Here is where a test for too many iterations would go.

            // This section of the program inspects for
            // negligible elements in the s and e arrays.  On
            // completion the variables kase and k are set as follows.

            // kase = 1     if s(p) and e[k-1] are negligible and k<p
            // kase = 2     if s(k) is negligible and k<p
            // kase = 3     if e[k-1] is negligible, k<p, and
            //              s(k), ..., s(p) are not negligible (qr step).
            // kase = 4     if e(p-1) is negligible (convergence).

            for ( k = p - 2; k >= -1; k-- ) {
                if ( k == -1 ) {
                    break;
                }
                if ( abs( e[k] ) <=
                     tiny + eps * (abs( s[k] ) + abs( s[k + 1] )) ) {
                    e[k] = 0.0;
                    break;
                }
            }
            if ( k == p - 2 ) {
                kase = 4;
            }
            else {
                var ks;
                for ( ks = p - 1; ks >= k; ks-- ) {
                    if ( ks == k ) {
                        break;
                    }
                    t = (ks != p ? abs( e[ks] ) : 0.) +
                        (ks != k + 1 ? abs( e[ks - 1] ) : 0.);
                    if ( abs( s[ks] ) <= tiny + eps * t ) {
                        s[ks] = 0.0;
                        break;
                    }
                }
                if ( ks == k ) {
                    kase = 3;
                }
                else if ( ks == p - 1 ) {
                    kase = 1;
                }
                else {
                    kase = 2;
                    k = ks;
                }
            }
            k++;

            // Perform the task indicated by kase.

            switch( kase ) {

                // Deflate negligible s(p).

                case 1:
                {
                    f = e[p - 2];
                    e[p - 2] = 0.0;
                    for ( j = p - 2; j >= k; j-- ) {
                        t = hypot( s[j], f );
                        var cs = s[j] / t;
                        var sn = f / t;
                        s[j] = t;
                        if ( j != k ) {
                            f = -sn * e[j - 1];
                            e[j - 1] = cs * e[j - 1];
                        }
                        if ( wantv ) {
                            for ( i = 0; i < n; i++ ) {
                                t = cs * V[i * n + j] + sn * V[i * n + p - 1];
                                V[i * n + p - 1] = -sn * V[i * n + j] + cs * V[i * n + p - 1];
                                V[i * n + j] = t;
                            }
                        }
                    }
                }
                    break;

                // Split at negligible s(k).

                case 2:
                {
                    f = e[k - 1];
                    e[k - 1] = 0.0;
                    for ( j = k; j < p; j++ ) {
                        t = hypot( s[j], f );
                        cs = s[j] / t;
                        sn = f / t;
                        s[j] = t;
                        f = -sn * e[j];
                        e[j] = cs * e[j];
                        if ( wantu ) {
                            for ( i = 0; i < m; i++ ) {
                                t = cs * U[i * nu + j] + sn * U[i * nu + k - 1];
                                U[i * nu + k - 1] = -sn * U[i * nu + j] + cs * U[i * nu + k - 1];
                                U[i * nu + j] = t;
                            }
                        }
                    }
                }
                    break;

                // Perform one qr step.

                case 3:
                {

                    // Calculate the shift.

                    var scale = max( max( max( max(
                            abs( s[p - 1] ), abs( s[p - 2] ) ), abs( e[p - 2] ) ),
                                                    abs( s[k] ) ), abs( e[k] ) );
                    var sp = s[p - 1] / scale;
                    var spm1 = s[p - 2] / scale;
                    var epm1 = e[p - 2] / scale;
                    var sk = s[k] / scale;
                    var ek = e[k] / scale;
                    var b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
                    var c = (sp * epm1) * (sp * epm1);
                    var shift = 0.0;
                    if ( (b != 0.0) || (c != 0.0) ) {
                        shift = Math.sqrt( b * b + c );
                        if ( b < 0.0 ) {
                            shift = -shift;
                        }
                        shift = c / (b + shift);
                    }
                    f = (sk + sp) * (sk - sp) + shift;
                    var g = sk * ek;

                    // Chase zeros.

                    for ( j = k; j < p - 1; j++ ) {
                        t = hypot( f, g );
                        cs = f / t;
                        sn = g / t;
                        if ( j != k ) {
                            e[j - 1] = t;
                        }
                        f = cs * s[j] + sn * e[j];
                        e[j] = cs * e[j] - sn * s[j];
                        g = sn * s[j + 1];
                        s[j + 1] = cs * s[j + 1];
                        if ( wantv ) {
                            for ( i = 0; i < n; i++ ) {
                                t = cs * V[i * n + j] + sn * V[i * n + j + 1];
                                V[i * n + j + 1] = -sn * V[i * n + j] + cs * V[i * n + j + 1];
                                V[i * n + j] = t;
                            }
                        }
                        t = hypot( f, g );
                        cs = f / t;
                        sn = g / t;
                        s[j] = t;
                        f = cs * e[j] + sn * s[j + 1];
                        s[j + 1] = -sn * e[j] + cs * s[j + 1];
                        g = sn * e[j + 1];
                        e[j + 1] = cs * e[j + 1];
                        if ( wantu && (j < m - 1) ) {
                            for ( i = 0; i < m; i++ ) {
                                t = cs * U[i * nu + j] + sn * U[i * nu + j + 1];
                                U[i * nu + j + 1] = -sn * U[i * nu + j] + cs * U[i * nu + j + 1];
                                U[i * nu + j] = t;
                            }
                        }
                    }
                    e[p - 2] = f;
                    iter = iter + 1;
                }
                    break;

                // Convergence.

                case 4:
                {

                    // Make the singular values positive.

                    if ( s[k] <= 0.0 ) {
                        s[k] = (s[k] < 0.0 ? -s[k] : 0.0);
                        if ( wantv ) {
                            for ( i = 0; i <= pp; i++ ) {
                                V[i * n + k] = -V[i * n + k];
                            }
                        }
                    }

                    // Order the singular values.

                    while ( k < pp ) {
                        if ( s[k] >= s[k + 1] ) {
                            break;
                        }
                        t = s[k];
                        s[k] = s[k + 1];
                        s[k + 1] = t;
                        if ( wantv && (k < n - 1) ) {
                            for ( i = 0; i < n; i++ ) {
                                t = V[i * n + k + 1];
                                V[i * n + k + 1] = V[i * n + k];
                                V[i * n + k] = t;
                            }
                        }
                        if ( wantu && (k < m - 1) ) {
                            for ( i = 0; i < m; i++ ) {
                                t = U[i * nu + k + 1];
                                U[i * nu + k + 1] = U[i * nu + k];
                                U[i * nu + k] = t;
                            }
                        }
                        k++;
                    }
                    iter = 0;
                    p--;
                }
                    break;
            }
        }
    };

    var SingularValueDecomposition = phet.math.SingularValueDecomposition;

    SingularValueDecomposition.prototype = {
        constructor: SingularValueDecomposition,

        getU: function () {
            return new Matrix( this.m, Math.min( this.m + 1, this.n ), this.U, true ); // the "fast" flag added, since U is Float32Array
        },

        getV: function () {
            return new Matrix( this.n, this.n, this.V, true );
        },

        getSingularValues: function () {
            return this.s;
        },

        getS: function () {
            var result = new Matrix( this.n, this.n );
            for ( var i = 0; i < this.n; i++ ) {
                for ( var j = 0; j < this.n; j++ ) {
                    result.entries[result.index( i, j )] = 0.0;
                }
                result.entries[result.index( i, i )] = this.s[i];
            }
            return result;
        },

        norm2: function () {
            return this.s[0];
        },

        cond: function () {
            return this.s[0] / this.s[Math.min( this.m, this.n ) - 1];
        },

        rank: function () {
            // changed to 23 from 52 (bits of mantissa), since we are using floats here!
            var eps = Math.pow( 2.0, -23.0 );
            var tol = Math.max( this.m, this.n ) * this.s[0] * eps;
            var r = 0;
            for ( var i = 0; i < this.s.length; i++ ) {
                if ( this.s[i] > tol ) {
                    r++;
                }
            }
            return r;
        }
    }
})();


//// Copyright 2002-2012, University of Colorado
//
///**
//* SVD decomposition, based on Jama
//*
//* @author Jonathan Olson
//*/
//
//var phet = phet || {};
//phet.math = phet.math || {};
//
//// create a new scope
//(function () {
//    var Matrix = phet.math.Matrix;
//
//    phet.math.SingularValueDecomposition = function ( matrix ) {
//        this.matrix = matrix;
//
//        var Arg = matrix;
//
//        // Derived from LINPACK code.
//        // Initialize.
//        var A = Arg.getArrayCopy();
//        this.m = Arg.getRowDimension();
//        this.n = Arg.getColumnDimension();
//        var m = this.m;
//        var n = this.n;
//
//        /* Apparently the failing cases are only a proper subset of (m<n),
//         so let's not throw error.  Correct fix to come later?
//         if (m<n) {
//         throw new IllegalArgumentException("Jama SVD only works for m >= n"); }
//         */
//        nu = Math.min( m, n );
//        var nu = nu;
//        this.s = new Float32Array( Math.min( m + 1, n ) );
//        var s = this.s;
//        this.U = new Float32Array( m * nu );
//        var U = this.U;
//        this.V = new Float32Array( n * n );
//        var V = this.V;
//        var e = new Float32Array( n );
//        var work = new Float32Array( m );
//        var wantu = true;
//        var wantv = true;
//
//        var i, j, k, t, f;
//
//        // Reduce A to bidiagonal form, storing the diagonal elements
//        // in s and the super-diagonal elements in e.
//
//        var nct = Math.min( m - 1, n );
//        var nrt = Math.max( 0, Math.min( n - 2, m ) );
//        for ( k = 0; k < Math.max( nct, nrt ); k++ ) {
//            if ( k < nct ) {
//
//                // Compute the transformation for the k-th column and
//                // place the k-th diagonal in s[k].
//                // Compute 2-norm of k-th column without under/overflow.
//                s[k] = 0;
//                for ( i = k; i < m; i++ ) {
//                    s[k] = Matrix.hypot( s[k], A[i * n + k] );
//                }
//                if ( s[k] != 0.0 ) {
//                    if ( A[k * n + k] < 0.0 ) {
//                        s[k] = -s[k];
//                    }
//                    for ( i = k; i < m; i++ ) {
//                        A[i * n + k] /= s[k];
//                    }
//                    A[k * n + k] += 1.0;
//                }
//                s[k] = -s[k];
//            }
//            for ( j = k + 1; j < n; j++ ) {
//                if ( (k < nct) && (s[k] != 0.0) ) {
//
//                    // Apply the transformation.
//
//                    t = 0;
//                    for ( i = k; i < m; i++ ) {
//                        t += A[i * n + k] * A[i * n + j];
//                    }
//                    t = -t / A[k * n + k];
//                    for ( i = k; i < m; i++ ) {
//                        A[i * n + j] += t * A[i * n + k];
//                    }
//                }
//
//                // Place the k-th row of A into e for the
//                // subsequent calculation of the row transformation.
//
//                e[j] = A[k * n + j];
//            }
//            if ( wantu && (k < nct) ) {
//
//                // Place the transformation in U for subsequent back
//                // multiplication.
//
//                for ( i = k; i < m; i++ ) {
//                    U[i * nu + k] = A[i * n + k];
//                }
//            }
//            if ( k < nrt ) {
//
//                // Compute the k-th row transformation and place the
//                // k-th super-diagonal in e[k].
//                // Compute 2-norm without under/overflow.
//                e[k] = 0;
//                for ( i = k + 1; i < n; i++ ) {
//                    e[k] = Matrix.hypot( e[k], e[i] );
//                }
//                if ( e[k] != 0.0 ) {
//                    if ( e[k + 1] < 0.0 ) {
//                        e[k] = -e[k];
//                    }
//                    for ( i = k + 1; i < n; i++ ) {
//                        e[i] /= e[k];
//                    }
//                    e[k + 1] += 1.0;
//                }
//                e[k] = -e[k];
//                if ( (k + 1 < m) && (e[k] != 0.0) ) {
//
//                    // Apply the transformation.
//
//                    for ( i = k + 1; i < m; i++ ) {
//                        work[i] = 0.0;
//                    }
//                    for ( j = k + 1; j < n; j++ ) {
//                        for ( i = k + 1; i < m; i++ ) {
//                            work[i] += e[j] * A[i * n + j];
//                        }
//                    }
//                    for ( j = k + 1; j < n; j++ ) {
//                        t = -e[j] / e[k + 1];
//                        for ( i = k + 1; i < m; i++ ) {
//                            A[i * n + j] += t * work[i];
//                        }
//                    }
//                }
//                if ( wantv ) {
//
//                    // Place the transformation in V for subsequent
//                    // back multiplication.
//
//                    for ( i = k + 1; i < n; i++ ) {
//                        V[i * n + k] = e[i];
//                    }
//                }
//            }
//        }
//
//        // Set up the final bidiagonal matrix or order p.
//
//        var p = Math.min( n, m + 1 );
//        if ( nct < n ) {
//            s[nct] = A[nct * n + nct];
//        }
//        if ( m < p ) {
//            s[p - 1] = 0.0;
//        }
//        if ( nrt + 1 < p ) {
//            e[nrt] = A[nrt * n + p - 1];
//        }
//        e[p - 1] = 0.0;
//
//        // If required, generate U.
//
//        if ( wantu ) {
//            for ( j = nct; j < nu; j++ ) {
//                for ( i = 0; i < m; i++ ) {
//                    U[i * nu + j] = 0.0;
//                }
//                U[j * nu + j] = 1.0;
//            }
//            for ( k = nct - 1; k >= 0; k-- ) {
//                if ( s[k] != 0.0 ) {
//                    for ( j = k + 1; j < nu; j++ ) {
//                        t = 0;
//                        for ( i = k; i < m; i++ ) {
//                            t += U[i * nu + k] * U[i * nu + j];
//                        }
//                        t = -t / U[k * nu + k];
//                        for ( i = k; i < m; i++ ) {
//                            U[i * nu + j] += t * U[i * nu + k];
//                        }
//                    }
//                    for ( i = k; i < m; i++ ) {
//                        U[i * nu + k] = -U[i * nu + k];
//                    }
//                    U[k * nu + k] = 1.0 + U[k * nu + k];
//                    for ( i = 0; i < k - 1; i++ ) {
//                        U[i * nu + k] = 0.0;
//                    }
//                }
//                else {
//                    for ( i = 0; i < m; i++ ) {
//                        U[i * nu + k] = 0.0;
//                    }
//                    U[k * nu + k] = 1.0;
//                }
//            }
//        }
//
//        // If required, generate V.
//
//        if ( wantv ) {
//            for ( k = n - 1; k >= 0; k-- ) {
//                if ( (k < nrt) && (e[k] != 0.0) ) {
//                    for ( j = k + 1; j < nu; j++ ) {
//                        t = 0;
//                        for ( i = k + 1; i < n; i++ ) {
//                            t += V[i * n + k] * V[i * n + j];
//                        }
//                        t = -t / V[k + 1 * n + k];
//                        for ( i = k + 1; i < n; i++ ) {
//                            V[i * n + j] += t * V[i * n + k];
//                        }
//                    }
//                }
//                for ( i = 0; i < n; i++ ) {
//                    V[i * n + k] = 0.0;
//                }
//                V[k * n + k] = 1.0;
//            }
//        }
//
//        // Main iteration loop for the singular values.
//
//        var pp = p - 1;
//        var iter = 0;
//        var eps = Math.pow( 2.0, -52.0 );
//        var tiny = Math.pow( 2.0, -966.0 );
//        while ( p > 0 ) {
//            var kase;
//
//            // Here is where a test for too many iterations would go.
//
//            // This section of the program inspects for
//            // negligible elements in the s and e arrays.  On
//            // completion the variables kase and k are set as follows.
//
//            // kase = 1     if s(p) and e[k-1] are negligible and k<p
//            // kase = 2     if s(k) is negligible and k<p
//            // kase = 3     if e[k-1] is negligible, k<p, and
//            //              s(k), ..., s(p) are not negligible (qr step).
//            // kase = 4     if e(p-1) is negligible (convergence).
//
//            for ( k = p - 2; k >= -1; k-- ) {
//                if ( k == -1 ) {
//                    break;
//                }
//                if ( Math.abs( e[k] ) <=
//                     tiny + eps * (Math.abs( s[k] ) + Math.abs( s[k + 1] )) ) {
//                    e[k] = 0.0;
//                    break;
//                }
//            }
//            if ( k == p - 2 ) {
//                kase = 4;
//            }
//            else {
//                var ks;
//                for ( ks = p - 1; ks >= k; ks-- ) {
//                    if ( ks == k ) {
//                        break;
//                    }
//                    t = (ks != p ? Math.abs( e[ks] ) : 0.) +
//                        (ks != k + 1 ? Math.abs( e[ks - 1] ) : 0.);
//                    if ( Math.abs( s[ks] ) <= tiny + eps * t ) {
//                        s[ks] = 0.0;
//                        break;
//                    }
//                }
//                if ( ks == k ) {
//                    kase = 3;
//                }
//                else if ( ks == p - 1 ) {
//                    kase = 1;
//                }
//                else {
//                    kase = 2;
//                    k = ks;
//                }
//            }
//            k++;
//
//            // Perform the task indicated by kase.
//
//            switch( kase ) {
//
//                // Deflate negligible s(p).
//
//                case 1:
//                {
//                    f = e[p - 2];
//                    e[p - 2] = 0.0;
//                    for ( j = p - 2; j >= k; j-- ) {
//                        t = Matrix.hypot( s[j], f );
//                        var cs = s[j] / t;
//                        var sn = f / t;
//                        s[j] = t;
//                        if ( j != k ) {
//                            f = -sn * e[j - 1];
//                            e[j - 1] = cs * e[j - 1];
//                        }
//                        if ( wantv ) {
//                            for ( i = 0; i < n; i++ ) {
//                                t = cs * V[i * n + j] + sn * V[i * n + p - 1];
//                                V[i * n + p - 1] = -sn * V[i * n + j] + cs * V[i * n + p - 1];
//                                V[i * n + j] = t;
//                            }
//                        }
//                    }
//                }
//                    break;
//
//                // Split at negligible s(k).
//
//                case 2:
//                {
//                    f = e[k - 1];
//                    e[k - 1] = 0.0;
//                    for ( j = k; j < p; j++ ) {
//                        t = Matrix.hypot( s[j], f );
//                        cs = s[j] / t;
//                        sn = f / t;
//                        s[j] = t;
//                        f = -sn * e[j];
//                        e[j] = cs * e[j];
//                        if ( wantu ) {
//                            for ( i = 0; i < m; i++ ) {
//                                t = cs * U[i * nu + j] + sn * U[i * nu + k - 1];
//                                U[i * nu + k - 1] = -sn * U[i * nu + j] + cs * U[i * nu + k - 1];
//                                U[i * nu + j] = t;
//                            }
//                        }
//                    }
//                }
//                    break;
//
//                // Perform one qr step.
//
//                case 3:
//                {
//
//                    // Calculate the shift.
//
//                    var scale = Math.max( Math.max( Math.max( Math.max(
//                            Math.abs( s[p - 1] ), Math.abs( s[p - 2] ) ), Math.abs( e[p - 2] ) ),
//                                                    Math.abs( s[k] ) ), Math.abs( e[k] ) );
//                    var sp = s[p - 1] / scale;
//                    var spm1 = s[p - 2] / scale;
//                    var epm1 = e[p - 2] / scale;
//                    var sk = s[k] / scale;
//                    var ek = e[k] / scale;
//                    var b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
//                    var c = (sp * epm1) * (sp * epm1);
//                    var shift = 0.0;
//                    if ( (b != 0.0) || (c != 0.0) ) {
//                        shift = Math.sqrt( b * b + c );
//                        if ( b < 0.0 ) {
//                            shift = -shift;
//                        }
//                        shift = c / (b + shift);
//                    }
//                    f = (sk + sp) * (sk - sp) + shift;
//                    var g = sk * ek;
//
//                    // Chase zeros.
//
//                    for ( j = k; j < p - 1; j++ ) {
//                        t = Matrix.hypot( f, g );
//                        cs = f / t;
//                        sn = g / t;
//                        if ( j != k ) {
//                            e[j - 1] = t;
//                        }
//                        f = cs * s[j] + sn * e[j];
//                        e[j] = cs * e[j] - sn * s[j];
//                        g = sn * s[j + 1];
//                        s[j + 1] = cs * s[j + 1];
//                        if ( wantv ) {
//                            for ( i = 0; i < n; i++ ) {
//                                t = cs * V[i * n + j] + sn * V[i * n + j + 1];
//                                V[i * n + j + 1] = -sn * V[i * n + j] + cs * V[i * n + j + 1];
//                                V[i * n + j] = t;
//                            }
//                        }
//                        t = Matrix.hypot( f, g );
//                        cs = f / t;
//                        sn = g / t;
//                        s[j] = t;
//                        f = cs * e[j] + sn * s[j + 1];
//                        s[j + 1] = -sn * e[j] + cs * s[j + 1];
//                        g = sn * e[j + 1];
//                        e[j + 1] = cs * e[j + 1];
//                        if ( wantu && (j < m - 1) ) {
//                            for ( i = 0; i < m; i++ ) {
//                                t = cs * U[i * nu + j] + sn * U[i * nu + j + 1];
//                                U[i * nu + j + 1] = -sn * U[i * nu + j] + cs * U[i * nu + j + 1];
//                                U[i * nu + j] = t;
//                            }
//                        }
//                    }
//                    e[p - 2] = f;
//                    iter = iter + 1;
//                }
//                    break;
//
//                // Convergence.
//
//                case 4:
//                {
//
//                    // Make the singular values positive.
//
//                    if ( s[k] <= 0.0 ) {
//                        s[k] = (s[k] < 0.0 ? -s[k] : 0.0);
//                        if ( wantv ) {
//                            for ( i = 0; i <= pp; i++ ) {
//                                V[i * n + k] = -V[i * n + k];
//                            }
//                        }
//                    }
//
//                    // Order the singular values.
//
//                    while ( k < pp ) {
//                        if ( s[k] >= s[k + 1] ) {
//                            break;
//                        }
//                        t = s[k];
//                        s[k] = s[k + 1];
//                        s[k + 1] = t;
//                        if ( wantv && (k < n - 1) ) {
//                            for ( i = 0; i < n; i++ ) {
//                                t = V[i * n + k + 1];
//                                V[i * n + k + 1] = V[i * n + k];
//                                V[i * n + k] = t;
//                            }
//                        }
//                        if ( wantu && (k < m - 1) ) {
//                            for ( i = 0; i < m; i++ ) {
//                                t = U[i * nu + k + 1];
//                                U[i * nu + k + 1] = U[i * nu + k];
//                                U[i * nu + k] = t;
//                            }
//                        }
//                        k++;
//                    }
//                    iter = 0;
//                    p--;
//                }
//                    break;
//            }
//        }
//    };
//
//    var SingularValueDecomposition = phet.math.SingularValueDecomposition;
//
//    SingularValueDecomposition.prototype = {
//        constructor: SingularValueDecomposition,
//
//        uIndex: function ( i, j ) {
//            return i * nu + j;
//        },
//
//        vIndex: function ( i, j ) {
//            return i * this.n + j;
//        },
//
//        getU: function () {
//            return new Matrix( this.m, Math.min( this.m + 1, this.n ), this.U, true ); // the "fast" flag added, since U is Float32Array
//        },
//
//        getV: function () {
//            return new Matrix( this.n, this.n, this.V, true );
//        },
//
//        getSingularValues: function () {
//            return this.s;
//        },
//
//        getS: function () {
//            var result = new Matrix( this.n, this.n );
//            for ( var i = 0; i < this.n; i++ ) {
//                for ( var j = 0; j < this.n; j++ ) {
//                    result.entries[result.index( i, j )] = 0.0;
//                }
//                result.entries[result.index( i, i )] = this.s[i];
//            }
//            return result;
//        },
//
//        norm2: function () {
//            return this.s[0];
//        },
//
//        cond: function () {
//            return this.s[0] / this.s[Math.min( this.m, this.n ) - 1];
//        },
//
//        rank: function () {
//            // changed to 23 from 52 (bits of mantissa), since we are using floats here!
//            var eps = Math.pow( 2.0, -23.0 );
//            var tol = Math.max( this.m, this.n ) * this.s[0] * eps;
//            var r = 0;
//            for ( var i = 0; i < this.s.length; i++ ) {
//                if ( this.s[i] > tol ) {
//                    r++;
//                }
//            }
//            return r;
//        }
//    }
//})();