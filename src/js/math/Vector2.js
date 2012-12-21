// Copyright 2002-2012, University of Colorado

/**
 * Basic 2-dimensional vector
 *
 * @author Jonathan Olson
 */

// ensure proper namespace
var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {

    phet.math.Vector2 = function ( x, y ) {
        // allow optional parameters
        this.x = x || 0;
        this.y = y || 0;
    };

    // shortcut within the scope
    var Vector2 = phet.math.Vector2;

    Vector2.createPolar = function ( magnitude, angle ) {
        return new Vector2( Math.cos( angle ), Math.sin( angle ) ).timesScalar( magnitude );
    };

    phet.math.Vector2.prototype = {
        constructor: phet.math.Vector2,

        magnitude: function () {
            return Math.sqrt( this.magnitudeSquared() );
        },

        magnitudeSquared: function () {
            this.dot( this );
        },

        dot: function ( v ) {
            return this.x * v.x + this.y * v.y;
        },

        /*---------------------------------------------------------------------------*
         * Immutables
         *----------------------------------------------------------------------------*/

        crossScalar: function ( v ) {
            return this.magnitude() * v.magnitude() * Math.sin( this.getAngle() - v.getAngle() );
        },

        normalized: function () {
            var mag = this.magnitude();
            if ( mag == 0 ) {
                throw new Error( "Cannot normalize a zero-magnitude vector" );
            }
            else {
                return new Vector2( this.x / mag, this.y / mag );
            }
        },

        timesScalar: function ( scalar ) {
            return new Vector2( this.x * scalar, this.y * scalar );
        },

        times: function( scalar ) {
            // make sure it's not a vector!
            phet.assert( scalar.dimension === undefined );
            return this.timesScalar( scalar );
        },

        componentTimes: function ( v ) {
            return new Vector2( this.x * v.x, this.y * v.y );
        },

        plus: function ( v ) {
            return new Vector2( this.x + v.x, this.y + v.y );
        },

        plusScalar: function ( scalar ) {
            return new Vector2( this.x + scalar, this.y + scalar );
        },

        minus: function ( v ) {
            return new Vector2( this.x - v.x, this.y - v.y );
        },

        minusScalar: function ( scalar ) {
            return new Vector2( this.x - scalar, this.y - scalar );
        },

        dividedScalar: function ( scalar ) {
            return new Vector2( this.x / scalar, this.y / scalar );
        },

        negated: function () {
            return new Vector2( -this.x, -this.y );
        },

        angle: function () {
            // TODO: verify this opposite ordering of x and y compared to Java
            return Math.atan2( this.x, this.y );
        },

        perpendicular: function () {
            return new Vector2( this.y, -this.x );
        },

        angleBetween: function ( v ) {
            return Math.acos( phet.math.clamp( this.normalized().dot( v.normalized() ), -1, 1 ) );
        },


        rotated: function ( angle ) {
            return Vector2.createPolar( this.magnitude(), this.getAngle + angle );
        },

        toString: function () {
            return "Vector2(" + this.x + ", " + this.y + ")";
        },

        toVector3: function () {
            return new phet.math.Vector3( this.x, this.y );
        },

        /*---------------------------------------------------------------------------*
         * Mutables
         *----------------------------------------------------------------------------*/

        set: function ( x, y ) {
            this.x = x;
            this.y = y;
        },

        setX: function ( x ) {
            this.x = x;
        },

        setY: function ( y ) {
            this.y = y;
        },

        copy: function ( v ) {
            this.x = v.x;
            this.y = v.y;
        },

        add: function ( v ) {
            this.x += v.x;
            this.y += v.y;
        },

        addScalar: function ( scalar ) {
            this.x += scalar;
            this.y += scalar;
        },

        subtract: function ( v ) {
            this.x -= v.x;
            this.y -= v.y;
        },

        subtractScalar: function ( scalar ) {
            this.x -= scalar;
            this.y -= scalar;
        },

        componentMultiply: function ( v ) {
            this.x *= v.x;
            this.y *= v.y;
        },

        divideScalar: function ( scalar ) {
            this.x /= scalar;
            this.y /= scalar;
        },

        negate: function () {
            this.x = -this.x;
            this.y = -this.y;
        },

        equals: function ( other, epsilon ) {
            if ( !epsilon ) {
                epsilon = 0;
            }
            return Math.abs( this.x - other.x ) + Math.abs( this.y - other.y ) <= epsilon;
        },

        isVector2: true,

        dimension: 2

    };

    /*---------------------------------------------------------------------------*
     * Immutable Vector form
     *----------------------------------------------------------------------------*/
    Vector2.Immutable = function ( x, y ) {
        this.x = x || 0;
        this.y = y || 0;
    };
    var Immutable = Vector2.Immutable;

    Immutable.prototype = new phet.math.Vector2();
    Immutable.prototype.constructor = Immutable;

    // throw errors whenever a mutable method is called on our immutable vector
    Immutable.mutableOverrideHelper = function ( mutableFunctionName ) {
        Immutable.prototype[mutableFunctionName] = function () {
            throw new Error( "Cannot call mutable method '" + mutableFunctionName + "' on immutable Vector2" );
        }
    };

    // TODO: better way to handle this list?
    Immutable.mutableOverrideHelper( 'set' );
    Immutable.mutableOverrideHelper( 'setX' );
    Immutable.mutableOverrideHelper( 'setY' );
    Immutable.mutableOverrideHelper( 'copy' );
    Immutable.mutableOverrideHelper( 'add' );
    Immutable.mutableOverrideHelper( 'addScalar' );
    Immutable.mutableOverrideHelper( 'subtract' );
    Immutable.mutableOverrideHelper( 'subtractScalar' );
    Immutable.mutableOverrideHelper( 'componentMultiply' );
    Immutable.mutableOverrideHelper( 'divideScalar' );
    Immutable.mutableOverrideHelper( 'negate' );

    // helpful immutable constants
    Vector2.ZERO = new Immutable( 0, 0 );
    Vector2.X_UNIT = new Immutable( 1, 0 );
    Vector2.Y_UNIT = new Immutable( 0, 1 );
})();
