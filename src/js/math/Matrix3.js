// Copyright 2002-2012, University of Colorado

/**
 * 3-dimensional Matrix
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {
    "use strict";
    
    phet.math.Matrix3 = function ( v00, v01, v02, v10, v11, v12, v20, v21, v22, type ) {

        // entries stored in column-major format
        this.entries = new Array( 9 );

        this.rowMajor( v00 === undefined ? 1 : v00, v01 || 0, v02 || 0,
                       v10 || 0, v11 === undefined ? 1 : v11, v12 || 0,
                       v20 || 0, v21 || 0, v22 === undefined ? 1 : v22,
                       type );
    };

    phet.math.Matrix3.Types = {
        OTHER: 0, // default
        IDENTITY: 1,
        TRANSLATION_2D: 2,
        SCALING: 3

        // TODO: possibly add rotations
    };

    var Matrix3 = phet.math.Matrix3;
    var Vector3 = phet.math.Vector3;
    var Types = Matrix3.Types;

    Matrix3.identity = function () {
        return new Matrix3( 1, 0, 0,
                            0, 1, 0,
                            0, 0, 1,
                            Types.IDENTITY );
    };

    Matrix3.translation = function ( x, y ) {
        return new Matrix3( 1, 0, x,
                            0, 1, y,
                            0, 0, 1,
                            Types.TRANSLATION_2D );
    };

    Matrix3.translationFromVector = function ( v ) { return Matrix3.translation( v.x, v.y ); };

    Matrix3.scaling = function ( x, y ) {
        // allow using one parameter to scale everything
        y = y === undefined ? x : y;

        return new Matrix3( x, 0, 0,
                            0, y, 0,
                            0, 0, 1,
                            Types.SCALING );
    };

    // axis is a normalized Vector3, angle in radians.
    Matrix3.rotationAxisAngle = function ( axis, angle ) {
        var c = Math.cos( angle );
        var s = Math.sin( angle );
        var C = 1 - c;

        return new Matrix3( axis.x * axis.x * C + c, axis.x * axis.y * C - axis.z * s, axis.x * axis.z * C + axis.y * s,
                            axis.y * axis.x * C + axis.z * s, axis.y * axis.y * C + c, axis.y * axis.z * C - axis.x * s,
                            axis.z * axis.x * C - axis.y * s, axis.z * axis.y * C + axis.x * s, axis.z * axis.z * C + c,
                            Types.OTHER );
    };

    // TODO: add in rotation from quaternion, and from quat + translation

    Matrix3.rotationX = function ( angle ) {
        var c = Math.cos( angle );
        var s = Math.sin( angle );

        return new Matrix3( 1, 0, 0,
                            0, c, -s,
                            0, s, c,
                            Types.OTHER );
    };

    Matrix3.rotationY = function ( angle ) {
        var c = Math.cos( angle );
        var s = Math.sin( angle );

        return new Matrix3( c, 0, s,
                            0, 1, 0,
                            -s, 0, c,
                            Types.OTHER );
    };

    Matrix3.rotationZ = function ( angle ) {
        var c = Math.cos( angle );
        var s = Math.sin( angle );

        return new Matrix3( c, -s, 0,
                            s, c, 0,
                            0, 0, 1,
                            Types.OTHER );
    };
    
    // standard 2d rotation
    Matrix3.rotation2 = Matrix3.rotationZ;
    
    Matrix3.fromSVGMatrix = function ( svgMatrix ) {
        return new Matrix3( svgMatrix.a, svgMatrix.c, svgMatrix.e,
                            svgMatrix.b, svgMatrix.d, svgMatrix.f,
                            0, 0, 1,
                            Types.OTHER );
    };

    // a rotation matrix that rotates A to B, by rotating about the axis A.cross( B ) -- Shortest path. ideally should be unit vectors
    Matrix3.rotateAToB = function ( a, b ) {
        // see http://graphics.cs.brown.edu/~jfh/papers/Moller-EBA-1999/paper.pdf for information on this implementation
        var start = a;
        var end = b;

        var epsilon = 0.0001;

        var e, h, f;

        var v = start.cross( end );
        e = start.dot( end );
        f = ( e < 0 ) ? -e : e;

        // if "from" and "to" vectors are nearly parallel
        if ( f > 1.0 - epsilon ) {
            var c1, c2, c3;
            /* coefficients for later use */
            var i, j;

            var x = new Vector3(
                    ( start.x > 0.0 ) ? start.x : -start.x,
                    ( start.y > 0.0 ) ? start.y : -start.y,
                    ( start.z > 0.0 ) ? start.z : -start.z
            );

            if ( x.x < x.y ) {
                if ( x.x < x.z ) {
                    x = Vector3.X_UNIT;
                }
                else {
                    x = Vector3.Z_UNIT;
                }
            }
            else {
                if ( x.y < x.z ) {
                    x = Vector3.Y_UNIT;
                }
                else {
                    x = Vector3.Z_UNIT;
                }
            }

            var u = x.minus( start );
            v = x.minus( end );

            c1 = 2.0 / u.dot( u );
            c2 = 2.0 / v.dot( v );
            c3 = c1 * c2 * u.dot( v );

            return Matrix3.IDENTITY.plus( Matrix3.rowMajor(
                    -c1 * u.x * u.x - c2 * v.x * v.x + c3 * v.x * u.x,
                    -c1 * u.x * u.y - c2 * v.x * v.y + c3 * v.x * u.y,
                    -c1 * u.x * u.z - c2 * v.x * v.z + c3 * v.x * u.z,
                    -c1 * u.y * u.x - c2 * v.y * v.x + c3 * v.y * u.x,
                    -c1 * u.y * u.y - c2 * v.y * v.y + c3 * v.y * u.y,
                    -c1 * u.y * u.z - c2 * v.y * v.z + c3 * v.y * u.z,
                    -c1 * u.z * u.x - c2 * v.z * v.x + c3 * v.z * u.x,
                    -c1 * u.z * u.y - c2 * v.z * v.y + c3 * v.z * u.y,
                    -c1 * u.z * u.z - c2 * v.z * v.z + c3 * v.z * u.z
            ) );
        }
        else {
            // the most common case, unless "start"="end", or "start"=-"end"
            var hvx, hvz, hvxy, hvxz, hvyz;
            h = 1.0 / ( 1.0 + e );
            hvx = h * v.x;
            hvz = h * v.z;
            hvxy = hvx * v.y;
            hvxz = hvx * v.z;
            hvyz = hvz * v.y;

            return Matrix3.rowMajor(
                    e + hvx * v.x, hvxy - v.z, hvxz + v.y,
                    hvxy + v.z, e + h * v.y * v.y, hvyz - v.x,
                    hvxz - v.y, hvyz + v.x, e + hvz * v.z
            );
        }
    };

    Matrix3.prototype = {
        constructor: Matrix3,

        rowMajor: function ( v00, v01, v02, v10, v11, v12, v20, v21, v22, type ) {
            this.entries[0] = v00;
            this.entries[1] = v10;
            this.entries[2] = v20;
            this.entries[3] = v01;
            this.entries[4] = v11;
            this.entries[5] = v21;
            this.entries[6] = v02;
            this.entries[7] = v12;
            this.entries[8] = v22;
            this.type = type === undefined ? Types.OTHER : type;
        },

        columnMajor: function ( v00, v10, v20, v01, v11, v21, v02, v12, v22, type ) {
            this.rowMajor( v00, v01, v02, v10, v11, v12, v20, v21, v22, type );
        },

        // convenience getters. inline usages of these when performance is critical? TODO: test performance of inlining these, with / without closure compiler
        m00: function () { return this.entries[0]; },
        m01: function () { return this.entries[3]; },
        m02: function () { return this.entries[6]; },
        m10: function () { return this.entries[1]; },
        m11: function () { return this.entries[4]; },
        m12: function () { return this.entries[7]; },
        m20: function () { return this.entries[2]; },
        m21: function () { return this.entries[5]; },
        m22: function () { return this.entries[8]; },

        plus: function ( m ) {
            return new Matrix3(
                    this.m00() + m.m00(), this.m01() + m.m01(), this.m02() + m.m02(),
                    this.m10() + m.m10(), this.m11() + m.m11(), this.m12() + m.m12(),
                    this.m20() + m.m20(), this.m21() + m.m21(), this.m22() + m.m22()
            );
        },

        minus: function ( m ) {
            return new Matrix3(
                    this.m00() - m.m00(), this.m01() - m.m01(), this.m02() - m.m02(),
                    this.m10() - m.m10(), this.m11() - m.m11(), this.m12() - m.m12(),
                    this.m20() - m.m20(), this.m21() - m.m21(), this.m22() - m.m22()
            );
        },

        transposed: function () {
            return new Matrix3( this.m00(), this.m10(), this.m20(),
                                this.m01(), this.m11(), this.m21(),
                                this.m02(), this.m12(), this.m22() );
        },

        negated: function () {
            return new Matrix3( -this.m00(), -this.m01(), -this.m02(),
                                -this.m10(), -this.m11(), -this.m12(),
                                -this.m20(), -this.m21(), -this.m22() );
        },

        inverted: function () {
            // TODO: optimizations for matrix types (like identity)

            var det = this.determinant();

            if ( det !== 0 ) {
                return new Matrix3(
                        ( -this.m12() * this.m21() + this.m11() * this.m22() ) / det,
                        ( this.m02() * this.m21() - this.m01() * this.m22() ) / det,
                        ( -this.m02() * this.m11() + this.m01() * this.m12() ) / det,
                        ( this.m12() * this.m20() - this.m10() * this.m22() ) / det,
                        ( -this.m02() * this.m20() + this.m00() * this.m22() ) / det,
                        ( this.m02() * this.m10() - this.m00() * this.m12() ) / det,
                        ( -this.m11() * this.m20() + this.m10() * this.m21() ) / det,
                        ( this.m01() * this.m20() - this.m00() * this.m21() ) / det,
                        ( -this.m01() * this.m10() + this.m00() * this.m11() ) / det
                );
            }
            else {
                throw new Error( "Matrix could not be inverted, determinant == 0" );
            }
        },

        timesMatrix: function ( m ) {
            var newType = Types.OTHER;
            if ( this.type == Types.TRANSLATION_2D && m.type == Types.TRANSLATION_2D ) {
                newType = Types.TRANSLATION_2D;
            }
            if ( this.type == Types.SCALING && m.type == Types.SCALING ) {
                newType = Types.SCALING;
            }
            if ( this.type == Types.IDENTITY ) {
                newType = m.type;
            }
            if ( m.type == Types.IDENTITY ) {
                newType = this.type;
            }
            return new Matrix3( this.m00() * m.m00() + this.m01() * m.m10() + this.m02() * m.m20(),
                                this.m00() * m.m01() + this.m01() * m.m11() + this.m02() * m.m21(),
                                this.m00() * m.m02() + this.m01() * m.m12() + this.m02() * m.m22(),
                                this.m10() * m.m00() + this.m11() * m.m10() + this.m12() * m.m20(),
                                this.m10() * m.m01() + this.m11() * m.m11() + this.m12() * m.m21(),
                                this.m10() * m.m02() + this.m11() * m.m12() + this.m12() * m.m22(),
                                this.m20() * m.m00() + this.m21() * m.m10() + this.m22() * m.m20(),
                                this.m20() * m.m01() + this.m21() * m.m11() + this.m22() * m.m21(),
                                this.m20() * m.m02() + this.m21() * m.m12() + this.m22() * m.m22(),
                                newType );
        },

        timesVector2: function( v ) {
            var x = this.m00() * v.x + this.m01() * v.y + this.m02();
            var y = this.m10() * v.x + this.m11() * v.y + this.m12();
            return new phet.math.Vector2( x, y );
        },

        timesVector3: function ( v ) {
            var x = this.m00() * v.x + this.m01() * v.y + this.m02() * v.z;
            var y = this.m10() * v.x + this.m11() * v.y + this.m12() * v.z;
            var z = this.m20() * v.x + this.m21() * v.y + this.m22() * v.z;
            return new phet.math.Vector3( x, y, z );
        },

        timesTransposeVector2: function ( v ) {
            var x = this.m00() * v.x + this.m10() * v.y;
            var y = this.m01() * v.x + this.m11() * v.y;
            return new phet.math.Vector2( x, y );
        },

        timesRelativeVector2: function ( v ) {
            var x = this.m00() * v.x + this.m10() * v.y;
            var y = this.m01() * v.y + this.m11() * v.y;
            return new phet.math.Vector2( x, y );
        },

        determinant: function () {
            return this.m00() * this.m11() * this.m22() + this.m01() * this.m12() * this.m20() + this.m02() * this.m10() * this.m21() - this.m02() * this.m11() * this.m20() - this.m01() * this.m10() * this.m22() - this.m00() * this.m12() * this.m21();
        },

        toString: function () {
            return this.m00() + " " + this.m01() + " " + this.m02() + "\n" +
                   this.m10() + " " + this.m11() + " " + this.m12() + "\n" +
                   this.m20() + " " + this.m21() + " " + this.m22();
        },

        toMatrix4: function () {
            return new phet.math.Matrix4( this.m00(), this.m01(), this.m02(), 0,
                                          this.m10(), this.m11(), this.m12(), 0,
                                          this.m20(), this.m21(), this.m22(), 0,
                                          0, 0, 0, 1 );
        },

        translation: function () { return new phet.math.Vector2( this.m02(), this.m12() ); },
        scaling: function () { return new phet.math.Vector3( this.m00(), this.m11(), this.m22() );},
        
        // angle in radians for the 2d rotation from this matrix, between pi, -pi
        rotation: function() {
            var transformedVector = this.timesVector2( phet.math.Vector2.X_UNIT ).minus( this.timesVector2( phet.math.Vector2.ZERO ) );
            return Math.atan2( transformedVector.y, transformedVector.x );
        },
        
        makeImmutable: function () {
            this.rowMajor = function () {
                throw new Error( "Cannot modify immutable matrix" );
            };
        },
        
        toSVGMatrix: function () {
            var result = document.createElementNS( 'http://www.w3.org/2000/svg', 'svg' ).createSVGMatrix();
            
            // top two rows
            result.a = this.m00();
            result.b = this.m10();
            result.c = this.m01();
            result.d = this.m11();
            result.e = this.m02();
            result.f = this.m12();
            
            return result;
        },
        
        // sets the transform of a Canvas 2D rendering context to the affine part of this matrix
        canvasSetTransform: function( context ) {
            context.setTransform(
                // inlined array entries
                this.entries[0],
                this.entries[1],
                this.entries[3],
                this.entries[4],
                this.entries[6],
                this.entries[7]
            );
        },
        
        // appends the affine part of this matrix to the Canvas 2D rendering context
        canvasAppendTransform: function( context ) {
            context.transform(
                // inlined array entries
                this.entries[0],
                this.entries[1],
                this.entries[3],
                this.entries[4],
                this.entries[6],
                this.entries[7]
            );
        },
        
        cssTransform: function() {
            // we need to prevent the numbers from being in an exponential toString form, since the CSS transform does not support that
            function cssNumber( number ) {
                // largest guaranteed number of digits according to https://developer.mozilla.org/en-US/docs/JavaScript/Reference/Global_Objects/Number/toFixed
                return number.toFixed( 20 );
            }
            // the inner part of a CSS3 transform, but remember to add the browser-specific parts!
            // TODO: do we need 'px' units on the last two (transform) attributes?
            return 'matrix(' + cssNumber( this.entries[0] ) + ',' + cssNumber( this.entries[1] ) + ',' + cssNumber( this.entries[3] ) + ',' + cssNumber( this.entries[4] ) + ',' + cssNumber( this.entries[6] ) + ',' + cssNumber( this.entries[7] ) + ')';
        }
    };

    // create an immutable
    Matrix3.IDENTITY = new Matrix3( 1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1,
                                    Types.IDENTITY );
    Matrix3.IDENTITY.makeImmutable();

})();