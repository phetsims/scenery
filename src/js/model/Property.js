// Copyright 2002-2012, University of Colorado

/**
 * See Java docs for now
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.model = phet.model || {};

// create a new scope
(function () {
    phet.model.Property = function ( value ) {
        this.observers = [];
        this.value = value;
        this.initialValue = value;
    };

    var Property = phet.model.Property;

    Property.prototype = {
        constructor: Property,

        addObserver: function ( observer, notifyOnAdd ) {
            this.observers.push( observer );
            if ( notifyOnAdd ) {
                observer.call( undefined, this.value, this.value );
            }
        },

        removeObserver: function ( observer ) {
            this.observers.splice( this.observers.indexOf( observer ), 1 );
        },

        notifyObservers: function ( oldValue, newValue ) {
            for ( var i in this.observers ) {
                if ( this.observers.hasOwnProperty( i ) ) {
                    this.observers[i].call( undefined, oldValue, newValue );
                }
            }
        },

        toString: function () {
            return this.value.toString();
        },

        get: function () {
            return this.value;
        },

        set: function ( value ) {
            var oldValue = this.value;
            var changed = value === null ? oldValue !== null : (value !== oldValue || (value.equals && value.equals( oldValue )));

            if ( changed ) {
                this.value = value;

                this.notifyObservers( oldValue, this.value );
            }
        },

        getInitialValue: function () {
            return this.initialValue;
        }
    };
})();
