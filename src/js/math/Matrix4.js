// Copyright 2002-2012, University of Colorado

/**
 * 4-dimensional Matrix
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {
  "use strict";
  
  var Float32Array = phet.Float32Array;

  phet.math.Matrix4 = function ( v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33, type ) {

    // entries stored in column-major format
    this.entries = new Float32Array( 16 );

    this.rowMajor( v00 === undefined ? 1 : v00, v01 || 0, v02 || 0, v03 || 0,
             v10 || 0, v11 === undefined ? 1 : v11, v12 || 0, v13 || 0,
             v20 || 0, v21 || 0, v22 === undefined ? 1 : v22, v23 || 0,
             v30 || 0, v31 || 0, v32 || 0, v33 === undefined ? 1 : v33,
             type );
  };

  phet.math.Matrix4.Types = {
    OTHER: 0, // default
    IDENTITY: 1,
    TRANSLATION_3D: 2,
    SCALING: 3

    // TODO: possibly add rotations
  };

  var Matrix4 = phet.math.Matrix4;
  var Types = Matrix4.Types;

  Matrix4.identity = function () {
    return new Matrix4( 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1,
              Types.IDENTITY );
  };

  Matrix4.translation = function ( x, y, z ) {
    return new Matrix4( 1, 0, 0, x,
              0, 1, 0, y,
              0, 0, 1, z,
              0, 0, 0, 1,
              Types.TRANSLATION_3D );
  };

  Matrix4.translationFromVector = function ( v ) { return Matrix4.translation( v.x, v.y, v.z ); };

  Matrix4.scaling = function ( x, y, z ) {
    // allow using one parameter to scale everything
    y = y === undefined ? x : y;
    z = z === undefined ? x : z;

    return new Matrix4( x, 0, 0, 0,
              0, y, 0, 0,
              0, 0, z, 0,
              0, 0, 0, 1,
              Types.SCALING );
  };

  // axis is a normalized Vector3, angle in radians.
  Matrix4.rotationAxisAngle = function ( axis, angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );
    var C = 1 - c;

    return new Matrix4( axis.x * axis.x * C + c, axis.x * axis.y * C - axis.z * s, axis.x * axis.z * C + axis.y * s, 0,
              axis.y * axis.x * C + axis.z * s, axis.y * axis.y * C + c, axis.y * axis.z * C - axis.x * s, 0,
              axis.z * axis.x * C - axis.y * s, axis.z * axis.y * C + axis.x * s, axis.z * axis.z * C + c, 0,
              0, 0, 0, 1,
              Types.OTHER );
  };

  // TODO: add in rotation from quaternion, and from quat + translation

  Matrix4.rotationX = function ( angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );

    return new Matrix4( 1, 0, 0, 0,
              0, c, -s, 0,
              0, s, c, 0,
              0, 0, 0, 1,
              Types.OTHER );
  };

  Matrix4.rotationY = function ( angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );

    return new Matrix4( c, 0, s, 0,
              0, 1, 0, 0,
              -s, 0, c, 0,
              0, 0, 0, 1,
              Types.OTHER );
  };

  Matrix4.rotationZ = function ( angle ) {
    var c = Math.cos( angle );
    var s = Math.sin( angle );

    return new Matrix4( c, -s, 0, 0,
              s, c, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1,
              Types.OTHER );
  };

  // aspect === width / height
  Matrix4.gluPerspective = function ( fovYRadians, aspect, zNear, zFar ) {
    var cotangent = Math.cos( fovYRadians ) / Math.sin( fovYRadians );

    return new Matrix4( cotangent / aspect, 0, 0, 0,
              0, cotangent, 0, 0,
              0, 0, ( zFar + zNear ) / ( zNear - zFar ), ( 2 * zFar * zNear ) / ( zNear - zFar ),
              0, 0, -1, 0 );
  };

  Matrix4.prototype = {
    constructor: Matrix4,

    rowMajor: function ( v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33, type ) {
      this.entries[0] = v00;
      this.entries[1] = v10;
      this.entries[2] = v20;
      this.entries[3] = v30;
      this.entries[4] = v01;
      this.entries[5] = v11;
      this.entries[6] = v21;
      this.entries[7] = v31;
      this.entries[8] = v02;
      this.entries[9] = v12;
      this.entries[10] = v22;
      this.entries[11] = v32;
      this.entries[12] = v03;
      this.entries[13] = v13;
      this.entries[14] = v23;
      this.entries[15] = v33;
      this.type = type === undefined ? Types.OTHER : type;
    },

    columnMajor: function ( v00, v10, v20, v30, v01, v11, v21, v31, v02, v12, v22, v32, v03, v13, v23, v33, type ) {
      this.rowMajor( v00, v01, v02, v03, v10, v11, v12, v13, v20, v21, v22, v23, v30, v31, v32, v33, type );
    },

    // convenience getters. inline usages of these when performance is critical? TODO: test performance of inlining these, with / without closure compiler
    m00: function () { return this.entries[0]; },
    m01: function () { return this.entries[4]; },
    m02: function () { return this.entries[8]; },
    m03: function () { return this.entries[12]; },
    m10: function () { return this.entries[1]; },
    m11: function () { return this.entries[5]; },
    m12: function () { return this.entries[9]; },
    m13: function () { return this.entries[13]; },
    m20: function () { return this.entries[2]; },
    m21: function () { return this.entries[6]; },
    m22: function () { return this.entries[10]; },
    m23: function () { return this.entries[14]; },
    m30: function () { return this.entries[3]; },
    m31: function () { return this.entries[7]; },
    m32: function () { return this.entries[11]; },
    m33: function () { return this.entries[15]; },

    plus: function ( m ) {
      return new Matrix4(
          this.m00() + m.m00(), this.m01() + m.m01(), this.m02() + m.m02(), this.m03() + m.m03(),
          this.m10() + m.m10(), this.m11() + m.m11(), this.m12() + m.m12(), this.m13() + m.m13(),
          this.m20() + m.m20(), this.m21() + m.m21(), this.m22() + m.m22(), this.m23() + m.m23(),
          this.m30() + m.m30(), this.m31() + m.m31(), this.m32() + m.m32(), this.m33() + m.m33()
      );
    },

    minus: function ( m ) {
      return new Matrix4(
          this.m00() - m.m00(), this.m01() - m.m01(), this.m02() - m.m02(), this.m03() - m.m03(),
          this.m10() - m.m10(), this.m11() - m.m11(), this.m12() - m.m12(), this.m13() - m.m13(),
          this.m20() - m.m20(), this.m21() - m.m21(), this.m22() - m.m22(), this.m23() - m.m23(),
          this.m30() - m.m30(), this.m31() - m.m31(), this.m32() - m.m32(), this.m33() - m.m33()
      );
    },

    transposed: function () {
      return new Matrix4( this.m00(), this.m10(), this.m20(), this.m30(),
                this.m01(), this.m11(), this.m21(), this.m31(),
                this.m02(), this.m12(), this.m22(), this.m32(),
                this.m03(), this.m13(), this.m23(), this.m33() );
    },

    negated: function () {
      return new Matrix4( -this.m00(), -this.m01(), -this.m02(), -this.m03(),
                -this.m10(), -this.m11(), -this.m12(), -this.m13(),
                -this.m20(), -this.m21(), -this.m22(), -this.m23(),
                -this.m30(), -this.m31(), -this.m32(), -this.m33() );
    },

    inverted: function () {
      // TODO: optimizations for matrix types (like identity)

      var det = this.determinant();

      if ( det !== 0 ) {
        return new Matrix4(
            ( -this.m31() * this.m22() * this.m13() + this.m21() * this.m32() * this.m13() + this.m31() * this.m12() * this.m23() - this.m11() * this.m32() * this.m23() - this.m21() * this.m12() * this.m33() + this.m11() * this.m22() * this.m33() ) / det,
            ( this.m31() * this.m22() * this.m03() - this.m21() * this.m32() * this.m03() - this.m31() * this.m02() * this.m23() + this.m01() * this.m32() * this.m23() + this.m21() * this.m02() * this.m33() - this.m01() * this.m22() * this.m33() ) / det,
            ( -this.m31() * this.m12() * this.m03() + this.m11() * this.m32() * this.m03() + this.m31() * this.m02() * this.m13() - this.m01() * this.m32() * this.m13() - this.m11() * this.m02() * this.m33() + this.m01() * this.m12() * this.m33() ) / det,
            ( this.m21() * this.m12() * this.m03() - this.m11() * this.m22() * this.m03() - this.m21() * this.m02() * this.m13() + this.m01() * this.m22() * this.m13() + this.m11() * this.m02() * this.m23() - this.m01() * this.m12() * this.m23() ) / det,
            ( this.m30() * this.m22() * this.m13() - this.m20() * this.m32() * this.m13() - this.m30() * this.m12() * this.m23() + this.m10() * this.m32() * this.m23() + this.m20() * this.m12() * this.m33() - this.m10() * this.m22() * this.m33() ) / det,
            ( -this.m30() * this.m22() * this.m03() + this.m20() * this.m32() * this.m03() + this.m30() * this.m02() * this.m23() - this.m00() * this.m32() * this.m23() - this.m20() * this.m02() * this.m33() + this.m00() * this.m22() * this.m33() ) / det,
            ( this.m30() * this.m12() * this.m03() - this.m10() * this.m32() * this.m03() - this.m30() * this.m02() * this.m13() + this.m00() * this.m32() * this.m13() + this.m10() * this.m02() * this.m33() - this.m00() * this.m12() * this.m33() ) / det,
            ( -this.m20() * this.m12() * this.m03() + this.m10() * this.m22() * this.m03() + this.m20() * this.m02() * this.m13() - this.m00() * this.m22() * this.m13() - this.m10() * this.m02() * this.m23() + this.m00() * this.m12() * this.m23() ) / det,
            ( -this.m30() * this.m21() * this.m13() + this.m20() * this.m31() * this.m13() + this.m30() * this.m11() * this.m23() - this.m10() * this.m31() * this.m23() - this.m20() * this.m11() * this.m33() + this.m10() * this.m21() * this.m33() ) / det,
            ( this.m30() * this.m21() * this.m03() - this.m20() * this.m31() * this.m03() - this.m30() * this.m01() * this.m23() + this.m00() * this.m31() * this.m23() + this.m20() * this.m01() * this.m33() - this.m00() * this.m21() * this.m33() ) / det,
            ( -this.m30() * this.m11() * this.m03() + this.m10() * this.m31() * this.m03() + this.m30() * this.m01() * this.m13() - this.m00() * this.m31() * this.m13() - this.m10() * this.m01() * this.m33() + this.m00() * this.m11() * this.m33() ) / det,
            ( this.m20() * this.m11() * this.m03() - this.m10() * this.m21() * this.m03() - this.m20() * this.m01() * this.m13() + this.m00() * this.m21() * this.m13() + this.m10() * this.m01() * this.m23() - this.m00() * this.m11() * this.m23() ) / det,
            ( this.m30() * this.m21() * this.m12() - this.m20() * this.m31() * this.m12() - this.m30() * this.m11() * this.m22() + this.m10() * this.m31() * this.m22() + this.m20() * this.m11() * this.m32() - this.m10() * this.m21() * this.m32() ) / det,
            ( -this.m30() * this.m21() * this.m02() + this.m20() * this.m31() * this.m02() + this.m30() * this.m01() * this.m22() - this.m00() * this.m31() * this.m22() - this.m20() * this.m01() * this.m32() + this.m00() * this.m21() * this.m32() ) / det,
            ( this.m30() * this.m11() * this.m02() - this.m10() * this.m31() * this.m02() - this.m30() * this.m01() * this.m12() + this.m00() * this.m31() * this.m12() + this.m10() * this.m01() * this.m32() - this.m00() * this.m11() * this.m32() ) / det,
            ( -this.m20() * this.m11() * this.m02() + this.m10() * this.m21() * this.m02() + this.m20() * this.m01() * this.m12() - this.m00() * this.m21() * this.m12() - this.m10() * this.m01() * this.m22() + this.m00() * this.m11() * this.m22() ) / det
        );
      }
      else {
        throw new Error( "Matrix could not be inverted, determinant === 0" );
      }
    },

    timesMatrix: function ( m ) {
      var newType = Types.OTHER;
      if ( this.type === Types.TRANSLATION_3D && m.type === Types.TRANSLATION_3D ) {
        newType = Types.TRANSLATION_3D;
      }
      if ( this.type === Types.SCALING && m.type === Types.SCALING ) {
        newType = Types.SCALING;
      }
      if ( this.type === Types.IDENTITY ) {
        newType = m.type;
      }
      if ( m.type === Types.IDENTITY ) {
        newType = this.type;
      }
      return new Matrix4( this.m00() * m.m00() + this.m01() * m.m10() + this.m02() * m.m20() + this.m03() * m.m30(),
                this.m00() * m.m01() + this.m01() * m.m11() + this.m02() * m.m21() + this.m03() * m.m31(),
                this.m00() * m.m02() + this.m01() * m.m12() + this.m02() * m.m22() + this.m03() * m.m32(),
                this.m00() * m.m03() + this.m01() * m.m13() + this.m02() * m.m23() + this.m03() * m.m33(),
                this.m10() * m.m00() + this.m11() * m.m10() + this.m12() * m.m20() + this.m13() * m.m30(),
                this.m10() * m.m01() + this.m11() * m.m11() + this.m12() * m.m21() + this.m13() * m.m31(),
                this.m10() * m.m02() + this.m11() * m.m12() + this.m12() * m.m22() + this.m13() * m.m32(),
                this.m10() * m.m03() + this.m11() * m.m13() + this.m12() * m.m23() + this.m13() * m.m33(),
                this.m20() * m.m00() + this.m21() * m.m10() + this.m22() * m.m20() + this.m23() * m.m30(),
                this.m20() * m.m01() + this.m21() * m.m11() + this.m22() * m.m21() + this.m23() * m.m31(),
                this.m20() * m.m02() + this.m21() * m.m12() + this.m22() * m.m22() + this.m23() * m.m32(),
                this.m20() * m.m03() + this.m21() * m.m13() + this.m22() * m.m23() + this.m23() * m.m33(),
                this.m30() * m.m00() + this.m31() * m.m10() + this.m32() * m.m20() + this.m33() * m.m30(),
                this.m30() * m.m01() + this.m31() * m.m11() + this.m32() * m.m21() + this.m33() * m.m31(),
                this.m30() * m.m02() + this.m31() * m.m12() + this.m32() * m.m22() + this.m33() * m.m32(),
                this.m30() * m.m03() + this.m31() * m.m13() + this.m32() * m.m23() + this.m33() * m.m33(),
                newType );
    },

    timesVector4: function ( v ) {
      var x = this.m00() * v.x + this.m01() * v.y + this.m02() * v.z + this.m03() * v.w;
      var y = this.m10() * v.x + this.m11() * v.y + this.m12() * v.z + this.m13() * v.w;
      var z = this.m20() * v.x + this.m21() * v.y + this.m22() * v.z + this.m23() * v.w;
      var w = this.m30() * v.x + this.m31() * v.y + this.m32() * v.z + this.m33() * v.w;
      return new phet.math.Vector4( x, y, z, w );
    },

    timesVector3: function ( v ) {
      return this.timesVector4( v.toVector4() ).toVector3();
    },

    timesTransposeVector4: function ( v ) {
      var x = this.m00() * v.x + this.m10() * v.y + this.m20() * v.z + this.m30() * v.w;
      var y = this.m01() * v.x + this.m11() * v.y + this.m21() * v.z + this.m31() * v.w;
      var z = this.m02() * v.x + this.m12() * v.y + this.m22() * v.z + this.m32() * v.w;
      var w = this.m03() * v.x + this.m13() * v.y + this.m23() * v.z + this.m33() * v.w;
      return new phet.math.Vector4( x, y, z, w );
    },

    timesTransposeVector3: function ( v ) {
      return this.timesTransposeVector4( v.toVector4() ).toVector3();
    },

    timesRelativeVector3: function ( v ) {
      var x = this.m00() * v.x + this.m10() * v.y + this.m20() * v.z;
      var y = this.m01() * v.y + this.m11() * v.y + this.m21() * v.z;
      var z = this.m02() * v.z + this.m12() * v.y + this.m22() * v.z;
      return new phet.math.Vector3( x, y, z );
    },

    determinant: function () {
      return this.m03() * this.m12() * this.m21() * this.m30() -
          this.m02() * this.m13() * this.m21() * this.m30() -
          this.m03() * this.m11() * this.m22() * this.m30() +
          this.m01() * this.m13() * this.m22() * this.m30() +
          this.m02() * this.m11() * this.m23() * this.m30() -
          this.m01() * this.m12() * this.m23() * this.m30() -
          this.m03() * this.m12() * this.m20() * this.m31() +
          this.m02() * this.m13() * this.m20() * this.m31() +
          this.m03() * this.m10() * this.m22() * this.m31() -
          this.m00() * this.m13() * this.m22() * this.m31() -
          this.m02() * this.m10() * this.m23() * this.m31() +
          this.m00() * this.m12() * this.m23() * this.m31() +
          this.m03() * this.m11() * this.m20() * this.m32() -
          this.m01() * this.m13() * this.m20() * this.m32() -
          this.m03() * this.m10() * this.m21() * this.m32() +
          this.m00() * this.m13() * this.m21() * this.m32() +
          this.m01() * this.m10() * this.m23() * this.m32() -
          this.m00() * this.m11() * this.m23() * this.m32() -
          this.m02() * this.m11() * this.m20() * this.m33() +
          this.m01() * this.m12() * this.m20() * this.m33() +
          this.m02() * this.m10() * this.m21() * this.m33() -
          this.m00() * this.m12() * this.m21() * this.m33() -
          this.m01() * this.m10() * this.m22() * this.m33() +
          this.m00() * this.m11() * this.m22() * this.m33();
    },

    toString: function () {
      return this.m00() + " " + this.m01() + " " + this.m02() + " " + this.m03() + "\n" +
           this.m10() + " " + this.m11() + " " + this.m12() + " " + this.m13() + "\n" +
           this.m20() + " " + this.m21() + " " + this.m22() + " " + this.m23() + "\n" +
           this.m30() + " " + this.m31() + " " + this.m32() + " " + this.m33();
    },

    translation: function () { return new phet.math.Vector3( this.m03(), this.m13(), this.m23() ); },
    scaling: function () { return new phet.math.Vector3( this.m00(), this.m11(), this.m22() );},

    makeImmutable: function () {
      this.rowMajor = function () {
        throw new Error( "Cannot modify immutable matrix" );
      };
    }
  };

  // create an immutable
  Matrix4.IDENTITY = new Matrix4();
  Matrix4.IDENTITY.makeImmutable();

})();
