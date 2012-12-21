// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.canvas = phet.canvas || {};

(function () {
    phet.canvas.initCanvas = function ( canvas ) {
        // Initialize the variable context to null.
        var context = null;

        try {
            // Try to grab the standard context. If it fails, fallback to experimental.
            context = canvas.getContext( "2d" );
        }
        catch( e ) {}

        // If we don't have a canvas context, give up now
        if ( !context ) {
            // TODO: show a visual display
            throw new Error( "Unable to initialize HTML5 canvas. Your browser may not support it." );
        }

        return context;
    };

    // see http://developer.apple.com/library/safari/#documentation/AudioVideo/Conceptual/HTML-canvas-guide/SettingUptheCanvas/SettingUptheCanvas.html#//apple_ref/doc/uid/TP40010542-CH2-SW5
    phet.canvas.backingScale = function ( context ) {
        if ( 'devicePixelRatio' in window ) {
            if ( window.devicePixelRatio > 1 && context.webkitBackingStorePixelRatio < 2 ) {
                return window.devicePixelRatio;
            }
        }
        return 1;
    }
})();