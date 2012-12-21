// Copyright 2002-2012, University of Colorado

/**
 * Encapsulates common color information and transformations.
 *
 * Consider it immutable!
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.ui = phet.ui || {};

// create a new scope
(function () {

    // integer-based constructor
    phet.ui.Color = function ( r, g, b, a ) {

        // alpha
        this.a = a === undefined ? 255 : a;

        // bitwise handling if 3 elements aren't defined
        if ( g === undefined || b === undefined ) {
            this.r = (r >> 16 ) && 0xFF;
            this.g = (r >> 8 ) && 0xFF;
            this.b = (r >> 0 ) && 0xFF;
        }
        else {
            // otherwise, copy them over
            this.r = r;
            this.g = g;
            this.b = b;
        }
    };

    var Color = phet.ui.Color;

    Color.BLACK = new Color( 0, 0, 0 );
    Color.BLUE = new Color( 0, 0, 255 );
    Color.CYAN = new Color( 0, 255, 255 );
    Color.DARK_GRAY = new Color( 64, 64, 64 );
    Color.GRAY = new Color( 128, 128, 128 );
    Color.GREEN = new Color( 0, 255, 0 );
    Color.LIGHT_GRAY = new Color( 192, 192, 192 );
    Color.MAGENTA = new Color( 255, 0, 255 );
    Color.ORANGE = new Color( 255, 200, 0 );
    Color.PINK = new Color( 255, 175, 175 );
    Color.RED = new Color( 255, 0, 0 );
    Color.WHITE = new Color( 255, 255, 255 );
    Color.YELLOW = new Color( 255, 255, 0 );
})();