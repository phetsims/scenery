// Copyright 2002-2013, University of Colorado

/**
 * The main 'scenery' namespace object for the exported (non-Require.js) API. Used internally
 * since it prevents Require.js issues with circular dependencies.
 *
 * The returned scenery object namespace may be incomplete if not all modules are listed as
 * dependencies. Please use the 'main' module for that purpose if all of Scenery is desired.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  window.sceneryAssert = require( 'ASSERT/assert' )( 'scenery' );
  window.sceneryAssertExtra = require( 'ASSERT/assert' )( 'scenery.extra' );
  
  window.sceneryLayerLog = null;
  window.sceneryEventLog = null;
  window.sceneryAccessibilityLog = null;
  
  var scratchCanvas = document.createElement( 'canvas' );
  var scratchContext = scratchCanvas.getContext( '2d' );
  
  // will be filled in by other modules
  return {
    scratchCanvas: scratchCanvas,   // a canvas used for convenience functions (think of it as having arbitrary state)
    scratchContext: scratchContext, // a context used for convenience functions (think of it as having arbitrary state)
    
    enableLayerLogging: function() {
      window.sceneryLayerLog = function( ob ) { console.log( ob ); };
    },
  
    disableLayerLogging: function() {
      window.sceneryLayerLog = null;
    },
    
    enableEventLogging: function() {
      window.sceneryEventLog = function( ob ) { console.log( ob ); };
    },
  
    disableEventLogging: function() {
      window.sceneryEventLog = null;
    },
    
    enableAccessibilityLogging: function() {
      window.sceneryAccessibilityLog = function( ob ) { console.log( ob ); };
    },
  
    disableAccessibilityLogging: function() {
      window.sceneryAccessibilityLog = null;
    }
  };
} );
