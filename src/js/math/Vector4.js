// Copyright 2002-2012, University of Colorado

/**
 * Basic 4-dimensional vector
 *
 * @author Jonathan Olson
 */

// ensure proper namespace
var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {

    phet.math.Vector4 = function ( x, y, z, w ) {
        // allow optional parameters
        this.x = x || 0;
        this.y = y || 0;
        this.z = z || 0;
        this.w = w !== undefined ? w : 1; // since w could be zero!
    };

    // shortcut within the scope
    var Vector4 = phet.math.Vector4;

    phet.math.Vector4.prototype = {
        constructor: phet.math.Vector4,

        magnitude: function () {
            return Math.sqrt( this.magnitudeSquared() );
        },

        magnitudeSquared: function () {
            this.dot( this );
        },

        dot: function ( v ) {
            return this.x * v.x + this.y * v.y + this.z * v.z + this.w * v.w;
        },

        /*---------------------------------------------------------------------------*
         * Immutables
         *----------------------------------------------------------------------------*/

        normalized: function () {
            var mag = this.magnitude();
            if ( mag == 0 ) {
                throw new Error( "Cannot normalize a zero-magnitude vector" );
            }
            else {
                return new Vector4( this.x / mag, this.y / mag, this.z / mag, this.w / mag );
            }
        },

        timesScalar: function ( scalar ) {
            return new Vector4( this.x * scalar, this.y * scalar, this.z * scalar, this.w * scalar );
        },

        times: function( scalar ) {
            // make sure it's not a vector!
            phet.assert( scalar.dimension === undefined );
            return this.timesScalar( scalar );
        },

        componentTimes: function ( v ) {
            return new Vector4( this.x * v.x, this.y * v.y, this.z * v.z, this.w * v.w );
        },

        plus: function ( v ) {
            return new Vector4( this.x + v.x, this.y + v.y, this.z + v.z, this.w + v.w );
        },

        plusScalar: function ( scalar ) {
            return new Vector4( this.x + scalar, this.y + scalar, this.z + scalar, this.w + scalar );
        },

        minus: function ( v ) {
            return new Vector4( this.x - v.x, this.y - v.y, this.z - v.z, this.w - v.w );
        },

        minusScalar: function ( scalar ) {
            return new Vector4( this.x - scalar, this.y - scalar, this.z - scalar, this.w - scalar );
        },

        dividedScalar: function ( scalar ) {
            return new Vector4( this.x / scalar, this.y / scalar, this.z / scalar, this.w / scalar );
        },

        negated: function () {
            return new Vector4( -this.x, -this.y, -this.z, -this.w );
        },

        angleBetween: function ( v ) {
            return Math.acos( phet.math.clamp( this.normalized().dot( v.normalized() ), -1, 1 ) );
        },

        toString: function () {
            return "Vector4(" + this.x + ", " + this.y + ", " + this.z + ", " + this.w + ")";
        },

        toVector3: function () {
            return new phet.math.Vector3( this.x, this.y, this.z );
        },

        /*---------------------------------------------------------------------------*
         * Mutables
         *----------------------------------------------------------------------------*/

        set: function ( x, y, z, w ) {
            this.x = x;
            this.y = y;
            this.z = z;
            this.w = w;
        },

        setX: function ( x ) {
            this.x = x;
        },

        setY: function ( y ) {
            this.y = y;
        },

        setZ: function ( z ) {
            this.z = z;
        },

        setW: function ( w ) {
            this.w = w;
        },

        copy: function ( v ) {
            this.x = v.x;
            this.y = v.y;
            this.z = v.z;
            this.w = v.w;
        },

        add: function ( v ) {
            this.x += v.x;
            this.y += v.y;
            this.z += v.z;
            this.w += v.w;
        },

        addScalar: function ( scalar ) {
            this.x += scalar;
            this.y += scalar;
            this.z += scalar;
            this.w += scalar;
        },

        subtract: function ( v ) {
            this.x -= v.x;
            this.y -= v.y;
            this.z -= v.z;
            this.w -= v.w;
        },

        subtractScalar: function ( scalar ) {
            this.x -= scalar;
            this.y -= scalar;
            this.z -= scalar;
            this.w -= scalar;
        },

        componentMultiply: function ( v ) {
            this.x *= v.x;
            this.y *= v.y;
            this.z *= v.z;
            this.w *= v.w;
        },

        divideScalar: function ( scalar ) {
            this.x /= scalar;
            this.y /= scalar;
            this.z /= scalar;
            this.w /= scalar;
        },

        negate: function () {
            this.x = -this.x;
            this.y = -this.y;
            this.z = -this.z;
            this.w = -this.w;
        },

        equals: function ( other, epsilon ) {
            if ( !epsilon ) {
                epsilon = 0;
            }
            return Math.abs( this.x - other.x ) + Math.abs( this.y - other.y ) + Math.abs( this.z - other.z ) + Math.abs( this.w - other.w ) <= epsilon;
        },

        isVector4: true,

        dimension: 4

    };

    /*---------------------------------------------------------------------------*
     * Immutable Vector form
     *----------------------------------------------------------------------------*/
    Vector4.Immutable = function ( x, y, z, w ) {
        this.x = x || 0;
        this.y = y || 0;
        this.z = z || 0;
        this.w = w !== undefined ? w : 1;
    };
    var Immutable = Vector4.Immutable;

    Immutable.prototype = new phet.math.Vector4();
    Immutable.prototype.constructor = Immutable;

    // throw errors whenever a mutable method is called on our immutable vector
    Immutable.mutableOverrideHelper = function ( mutableFunctionName ) {
        Immutable.prototype[mutableFunctionName] = function () {
            throw new Error( "Cannot call mutable method '" + mutableFunctionName + "' on immutable Vector4" );
        }
    };

    // TODO: better way to handle this list?
    Immutable.mutableOverrideHelper( 'set' );
    Immutable.mutableOverrideHelper( 'setX' );
    Immutable.mutableOverrideHelper( 'setY' );
    Immutable.mutableOverrideHelper( 'setZ' );
    Immutable.mutableOverrideHelper( 'setW' );
    Immutable.mutableOverrideHelper( 'copy' );
    Immutable.mutableOverrideHelper( 'add' );
    Immutable.mutableOverrideHelper( 'addScalar' );
    Immutable.mutableOverrideHelper( 'subtract' );
    Immutable.mutableOverrideHelper( 'subtractScalar' );
    Immutable.mutableOverrideHelper( 'componentMultiply' );
    Immutable.mutableOverrideHelper( 'divideScalar' );
    Immutable.mutableOverrideHelper( 'negate' );

    // helpful immutable constants
    Vector4.ZERO = new Immutable( 0, 0, 0, 0 );
    Vector4.X_UNIT = new Immutable( 1, 0, 0, 0 );
    Vector4.Y_UNIT = new Immutable( 0, 1, 0, 0 );
    Vector4.Z_UNIT = new Immutable( 0, 0, 1, 0 );
    Vector4.W_UNIT = new Immutable( 0, 0, 0, 1 );
})();
