// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.math = phet.math || {};

phet.math.clamp = function ( value, min, max ) {
    if ( value < min ) {
        return min;
    }
    else if ( value > max ) {
        return max;
    }
    else {
        return value;
    }
};

phet.math.toRadians = function ( degrees ) {
    return Math.PI * degrees / 180.0;
};

