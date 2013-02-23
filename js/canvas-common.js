// Copyright 2002-2012, University of Colorado

define( function( require ) {
  "use strict";
  
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
  
  phet.canvas.backingStorePixelRatio = function( context ) {
    return context.webkitBackingStorePixelRatio ||
           context.mozBackingStorePixelRatio ||
           context.msBackingStorePixelRatio ||
           context.oBackingStorePixelRatio ||
           context.backingStorePixelRatio || 1;
  };
  
  // see http://developer.apple.com/library/safari/#documentation/AudioVideo/Conceptual/HTML-canvas-guide/SettingUptheCanvas/SettingUptheCanvas.html#//apple_ref/doc/uid/TP40010542-CH2-SW5
  // and updated based on http://www.html5rocks.com/en/tutorials/canvas/hidpi/
  phet.canvas.backingScale = function ( context ) {
    if ( 'devicePixelRatio' in window ) {
      var backingStoreRatio = phet.canvas.backingStorePixelRatio( context );
      
      return window.devicePixelRatio / backingStoreRatio;
    }
    return 1;
  };
} );
