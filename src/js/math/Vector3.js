// Copyright 2002-2012, University of Colorado

/**
 * Basic 3-dimensional vector
 *
 * @author Jonathan Olson
 */

// ensure proper namespace
var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {

    phet.math.Vector3 = function ( x, y, z ) {
        // allow optional parameters
        this.x = x || 0;
        this.y = y || 0;
        this.z = z || 0;
    };

    // shortcut within the scope
    var Vector3 = phet.math.Vector3;

    phet.math.Vector3.prototype = {
        constructor: phet.math.Vector3,

        magnitude: function () {
            return Math.sqrt( this.magnitudeSquared() );
        },

        magnitudeSquared: function () {
            return this.dot( this );
        },

        dot: function ( v ) {
            return this.x * v.x + this.y * v.y + this.z * v.z;
        },

        /*---------------------------------------------------------------------------*
         * Immutables
         *----------------------------------------------------------------------------*/

        cross: function ( v ) {
            return new Vector3(
                    this.y * v.z - this.z * v.y,
                    this.z * v.x - this.x * v.z,
                    this.x * v.y - this.y * v.x
            )
        },

        normalized: function () {
            var mag = this.magnitude();
            if ( mag == 0 ) {
                throw new Error( "Cannot normalize a zero-magnitude vector" );
            }
            else {
                return new Vector3( this.x / mag, this.y / mag, this.z / mag );
            }
        },

        timesScalar: function ( scalar ) {
            return new Vector3( this.x * scalar, this.y * scalar, this.z * scalar );
        },

        times: function( scalar ) {
            // make sure it's not a vector!
            phet.assert( scalar.dimension === undefined );
            return this.timesScalar( scalar );
        },

        componentTimes: function ( v ) {
            return new Vector3( this.x * v.x, this.y * v.y, this.z * v.z );
        },

        plus: function ( v ) {
            return new Vector3( this.x + v.x, this.y + v.y, this.z + v.z );
        },

        plusScalar: function ( scalar ) {
            return new Vector3( this.x + scalar, this.y + scalar, this.z + scalar );
        },

        minus: function ( v ) {
            return new Vector3( this.x - v.x, this.y - v.y, this.z - v.z );
        },

        minusScalar: function ( scalar ) {
            return new Vector3( this.x - scalar, this.y - scalar, this.z - scalar );
        },

        dividedScalar: function ( scalar ) {
            return new Vector3( this.x / scalar, this.y / scalar, this.z / scalar );
        },

        negated: function () {
            return new Vector3( -this.x, -this.y, -this.z );
        },

        angleBetween: function ( v ) {
            return Math.acos( phet.math.clamp( this.normalized().dot( v.normalized() ), -1, 1 ) );
        },

        toString: function () {
            return "Vector3(" + this.x + ", " + this.y + ", " + this.z + ")";
        },

        toVector2: function () {
            return new phet.math.Vector2( this.x, this.y );
        },

        toVector4: function () {
            return new phet.math.Vector4( this.x, this.y, this.z );
        },

        /*---------------------------------------------------------------------------*
         * Mutables
         *----------------------------------------------------------------------------*/

        set: function ( x, y, z ) {
            this.x = x;
            this.y = y;
            this.z = z;
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

        copy: function ( v ) {
            this.x = v.x;
            this.y = v.y;
            this.z = v.z;
        },

        add: function ( v ) {
            this.x += v.x;
            this.y += v.y;
            this.z += v.z;
        },

        addScalar: function ( scalar ) {
            this.x += scalar;
            this.y += scalar;
            this.z += scalar;
        },

        subtract: function ( v ) {
            this.x -= v.x;
            this.y -= v.y;
            this.z -= v.z;
        },

        subtractScalar: function ( scalar ) {
            this.x -= scalar;
            this.y -= scalar;
            this.z -= scalar;
        },

        componentMultiply: function ( v ) {
            this.x *= v.x;
            this.y *= v.y;
            this.z *= v.z;
        },

        divideScalar: function ( scalar ) {
            this.x /= scalar;
            this.y /= scalar;
            this.z /= scalar;
        },

        negate: function () {
            this.x = -this.x;
            this.y = -this.y;
            this.z = -this.z;
        },

        equals: function ( other, epsilon ) {
            if ( !epsilon ) {
                epsilon = 0;
            }
            return Math.abs( this.x - other.x ) + Math.abs( this.y - other.y ) + Math.abs( this.z - other.z ) <= epsilon;
        },

        isVector3: true,

        dimension: 3

    };

    /*---------------------------------------------------------------------------*
     * Immutable Vector form
     *----------------------------------------------------------------------------*/
    Vector3.Immutable = function ( x, y, z ) {
        this.x = x || 0;
        this.y = y || 0;
        this.z = z || 0;
    };
    var Immutable = Vector3.Immutable;

    Immutable.prototype = new phet.math.Vector3();
    Immutable.prototype.constructor = Immutable;

    // throw errors whenever a mutable method is called on our immutable vector
    Immutable.mutableOverrideHelper = function ( mutableFunctionName ) {
        Immutable.prototype[mutableFunctionName] = function () {
            throw new Error( "Cannot call mutable method '" + mutableFunctionName + "' on immutable Vector3" );
        }
    };

    // TODO: better way to handle this list?
    Immutable.mutableOverrideHelper( 'set' );
    Immutable.mutableOverrideHelper( 'setX' );
    Immutable.mutableOverrideHelper( 'setY' );
    Immutable.mutableOverrideHelper( 'setZ' );
    Immutable.mutableOverrideHelper( 'copy' );
    Immutable.mutableOverrideHelper( 'add' );
    Immutable.mutableOverrideHelper( 'addScalar' );
    Immutable.mutableOverrideHelper( 'subtract' );
    Immutable.mutableOverrideHelper( 'subtractScalar' );
    Immutable.mutableOverrideHelper( 'componentMultiply' );
    Immutable.mutableOverrideHelper( 'divideScalar' );
    Immutable.mutableOverrideHelper( 'negate' );

    // helpful immutable constants
    Vector3.ZERO = new Immutable( 0, 0, 0 );
    Vector3.X_UNIT = new Immutable( 1, 0, 0 );
    Vector3.Y_UNIT = new Immutable( 0, 1, 0 );
    Vector3.Z_UNIT = new Immutable( 0, 0, 1 );
})();
