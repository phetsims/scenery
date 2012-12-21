// Copyright 2002-2012, University of Colorado

/**
 * A group of electron pairs. The pairs may be part of a bond, or may be a lone electron pair.
 */

var phet = phet || {};
phet.moleculeshapes = phet.moleculeshapes || {};
phet.moleculeshapes.model = phet.moleculeshapes.model || {};

// create a new scope
(function () {
    var Property = phet.model.Property;
    var Vector3 = phet.math.Vector3;

    var nextId = 0;

    phet.moleculeshapes.model.PairGroup = function ( position, isLonePair, startDragged, element ) {
        // unique identifier
        this.id = nextId++;

        this.position = new Property( position );
        this.velocity = new Property( Vector3.ZERO );
        this.isLonePair = isLonePair;
        this.userControlled = new Property( startDragged );

        // undefined for vsepr pair groups
        this.element = element;

        this.position.addObserver( function ( oldValue, newValue ) {
//            if ( newValue.magnitude() > 40 ) {
//                throw new Error( "Magnitude way too large!" );
//            }
            if ( isNaN( newValue.x ) ) {
                throw new Error( "NaN detected in position!" );
            }
            if ( oldValue.equals( Vector3.ZERO ) ) {
                throw new Error( "center molecule position change?" );
            }
        } );

        this.velocity.addObserver( function ( oldValue, newValue ) {
            if ( isNaN( newValue.x ) ) {
                throw new Error( "NaN detected in velocity!" );
            }
        } );
    };

    var PairGroup = phet.moleculeshapes.model.PairGroup;

    /*---------------------------------------------------------------------------*
     * constants
     *----------------------------------------------------------------------------*/
    PairGroup.BONDED_PAIR_DISTANCE = 10.0;
    PairGroup.LONE_PAIR_DISTANCE = 7.0;

    PairGroup.ELECTRON_PAIR_REPULSION_SCALE = 30000;
    PairGroup.ANGLE_REPULSION_SCALE = 3;
    PairGroup.JITTER_SCALE = 0.001;
    PairGroup.DAMPING_FACTOR = 0.1;

    // TODO: this is horrible. refactor it!
    PairGroup.REAL_TMP_SCALE = 5.5; // TODO: deal with units correctly in the 1st tab model so we can remove this

    function interpolate( a, b, ratio ) {
        return a * ( 1 - ratio ) + b * ratio;
    }

    // Returns a unit vector that is the component of "vector" that is perpendicular to the "position" vector
    PairGroup.getTangentDirection = function ( position, vector ) {
        var normalizedPosition = position.normalized();
        return vector.minus( normalizedPosition.times( vector.dot( normalizedPosition ) ) );
    };

    // helps avoid oscillation when the frame-rate is low, due to how the damping is implemented
    PairGroup.getTimescaleImpulseFactor = function ( timeElapsed ) {
        return Math.sqrt( ( timeElapsed > 0.017 ) ? 0.017 / timeElapsed : 1 );
    };

    PairGroup.prototype = {
        constructor: PairGroup,

        attractToIdealDistance: function ( timeElapsed, oldDistance, bond ) {
            if ( this.userControlled.get() ) {
                // don't process if being dragged
                return;
            }
            var origin = bond.getOtherAtom( this ).position.get();

            var isTerminalLonePair = !origin.equals( Vector3.ZERO );

            var idealDistanceFromCenter = bond.length * PairGroup.REAL_TMP_SCALE;

            /*---------------------------------------------------------------------------*
             * prevent movement away from our ideal distance
             *----------------------------------------------------------------------------*/
            var currentError = Math.abs( ( this.position.get().minus( origin ) ).magnitude() - idealDistanceFromCenter );
            var oldError = Math.abs( oldDistance - idealDistanceFromCenter );
            if ( currentError > oldError ) {
                // our error is getting worse! for now, don't let us slide AWAY from the ideal distance ever
                // set our distance to the old one, so it is easier to process
                this.position.set( this.position.get().normalized().times( oldDistance ).plus( origin ) );
            }

            /*---------------------------------------------------------------------------*
             * use damped movement towards our ideal distance
             *----------------------------------------------------------------------------*/
            var toCenter = this.position.get().minus( origin );

            var distance = toCenter.magnitude();
            var directionToCenter = toCenter.normalized();

            var offset = idealDistanceFromCenter - distance;

            // just modify position for now so we don't get any oscillations
            var ratioOfMovement = Math.pow( 0.1, 0.016 / timeElapsed ); // scale this exponentially by how much time has elapsed, so the more time taken, the faster we move towards the ideal distance
            if ( isTerminalLonePair ) {
                ratioOfMovement = 1;
            }
            this.position.set( this.position.get().plus( directionToCenter.times( ratioOfMovement * offset ) ) );
        },

        /**
         * @param other                    The pair group whose force on this object we want
         * @param timeElapsed              Time elapsed (thus we return an impulse instead of a force)
         * @param trueLengthsRatioOverride From 0 to 1. If 0, lone pairs will behave the same as bonds. If 1, lone pair distance will be taken into account
         * @return Repulsion force on this pair group, from the other pair group
         */
        getRepulsionImpulse: function ( other, timeElapsed, trueLengthsRatioOverride ) {
            // only handle the force on this object for now

            /*---------------------------------------------------------------------------*
             * adjust the logical positions when the repulsion modifier is less than 1
             *
             * (this allows us to get the "VSEPR" correct geometry even with lone pairs.
             * since lone pairs are closer in, an actual Coulomb model would diverge from
             * the VSEPR model angles. Here, we converge to the model VSEPR behavior, but
             * allow correct Coulomb calculations at greater distances
             *----------------------------------------------------------------------------*/

            // adjusted distances from the center atom
            var adjustedMagnitude = interpolate( PairGroup.BONDED_PAIR_DISTANCE, this.position.get().magnitude(), trueLengthsRatioOverride );
            var adjustedOtherMagnitude = interpolate( PairGroup.BONDED_PAIR_DISTANCE, other.position.get().magnitude(), trueLengthsRatioOverride );

            // adjusted positions
            var adjustedPosition = this.position.get().normalized().times( adjustedMagnitude );
            var adjustedOtherPosition = other.position.get().magnitude() == 0 ? new Vector3() : other.position.get().normalized().times( adjustedOtherMagnitude );

            // from other => this (adjusted)
            var delta = adjustedPosition.minus( adjustedOtherPosition );

            /*---------------------------------------------------------------------------*
             * coulomb repulsion
             *----------------------------------------------------------------------------*/

            var repulsionFactor = 1;

            // mimic Coulomb's Law
            var coulombVelocityDelta = delta.normalized().times( timeElapsed * PairGroup.ELECTRON_PAIR_REPULSION_SCALE * repulsionFactor / ( delta.magnitude() * delta.magnitude() ) );

            // apply a nonphysical reduction on coulomb's law when the frame-rate is low, so we can avoid oscillation
            var coulombDowngrade = PairGroup.getTimescaleImpulseFactor( timeElapsed ); // TODO: isolate the "standard" tpf?
            return coulombVelocityDelta.times( coulombDowngrade );
        },

        repulseFrom: function ( other, timeElapsed, trueLengthsRatioOverride ) {
            this.addVelocity( this.getRepulsionImpulse( other, timeElapsed, trueLengthsRatioOverride ) );
        },

        addVelocity: function ( velocityChange ) {
            // don't allow velocity changes if we are dragging it, OR if it is an atom at the origin
            if ( !this.userControlled.get() && !this.isCentralAtom() ) {
                this.velocity.set( this.velocity.get().plus( velocityChange ) );
            }
        },

        getPushFactor: function () {
            return this.isLonePair ? 1 : 1;
        },

        estimatePush: function ( angle ) {
            var result = 10 / ( angle * angle );
            if ( isNaN( result ) ) {
                return 0;
            }
            return result;
        },

        stepForward: function ( timeElapsed ) {
            // velocity changes so that it doesn't point at all towards or away from the origin
            var velocityMagnitudeOutwards = this.velocity.get().dot( this.position.get().normalized() );
            if ( this.position.get().magnitude() > 0 ) {
                this.velocity.set( this.velocity.get().minus( this.position.get().normalized().times( velocityMagnitudeOutwards ) ) ); // subtract the outwards-component out
            }

            // move position forward by scaled velocity
            this.position.set( this.position.get().plus( this.velocity.get().times( timeElapsed ) ) );

            // add in damping so we don't get the kind of oscillation that we are seeing
            var damping = 1 - PairGroup.DAMPING_FACTOR;
            damping = Math.pow( damping, timeElapsed / 0.017 ); // based so that we have no modification at 0.017
            this.velocity.set( this.velocity.get().times( damping ) );
        },

        dragToPosition: function ( vector ) {
            this.position.set( vector );

            // stop any velocity that was moving the pair
            this.velocity.set( new Vector3() );
        },

        isCentralAtom: function () {
            return !this.isLonePair && this.position.get().equals( Vector3.ZERO );
        },

        getElement: function () {
            return this.element;
        }
    };
})();
