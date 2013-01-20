// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.scene = phet.scene || {};

(function(){
    
    var Matrix3 = phet.math.Matrix3;
    var Transform3 = phet.math.Transform3;
    
    // drawingStyles should include font, textAlign, textBaseline, direction
    // textAlign = 'left', textBaseline = 'alphabetic' and direction = 'ltr' are recommended
	phet.scene.canvasTextBoundsAccurate = function( text, fontDrawingStyles ) {
        
    };
    
    phet.scene.canvasAccurateBounds = function( renderToCanvas, options ) {
        // how close to the actual bounds do we need to be?
        var precision = ( options && options.precision ) ? options.precision : 0.001;
        
        // 512x512 default square resolution
        var resolution = ( options && options.resolution ) ? options.resolution : 512;
        
        // at 1/4x default, we want to be able to get the bounds accurately for something as large as 4x our initial resolution
        // divisible by 2 so hopefully we avoid more quirks from Canvas rendering engines
        var initialScale = ( options && options.initialScale ) ? options.initialScale : 0.25;
        
        var canvas = document.createElement( 'canvas' );
        canvas.width = resolution;
        canvas.height = resolution;
        var context = phet.canvas.initCanvas( canvas );
        
        
        
        
        scene.renderToCanvas( canvas, context );
        var data = context.getImageData( 0, 0, canvasWidth, canvasHeight );
    };
})();