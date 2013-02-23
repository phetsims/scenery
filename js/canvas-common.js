// Copyright 2002-2012, University of Colorado

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
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
