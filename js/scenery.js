// Copyright 2002-2014, University of Colorado

/**
 * The main 'scenery' namespace object for the exported (non-Require.js) API. Used internally
 * since it prevents Require.js issues with circular dependencies.
 *
 * The returned scenery object namespace may be incomplete if not all modules are listed as
 * dependencies. Please use the 'main' module for that purpose if all of Scenery is desired.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  window.assert = window.assert || require( 'ASSERT/assert' )( 'basic' );
  window.assertSlow = window.assertSlow || require( 'ASSERT/assert' )( 'slow', true );
  
  window.sceneryLog = null;
  window.sceneryEventLog = null;
  window.sceneryAccessibilityLog = null;
  
  // object allocation tracking
  window.phetAllocation = require( 'PHET_CORE/phetAllocation' );
 
   var scratchCanvas = document.createElement( 'canvas' );
  var scratchContext = scratchCanvas.getContext( '2d' );
  
  // will be filled in by other modules
  var scenery = {
    assert: assert,
    
    scratchCanvas: scratchCanvas,   // a canvas used for convenience functions (think of it as having arbitrary state)
    scratchContext: scratchContext, // a context used for convenience functions (think of it as having arbitrary state)
    
    svgns: 'http://www.w3.org/2000/svg',     // svg namespace
    xlinkns: 'http://www.w3.org/1999/xlink', // x-link namespace
    
    logString: '',
    
    logFunction: function() {
      // allow for the console to not exist
      window.console && window.console.log && window.console.log.apply( window.console, Array.prototype.slice.call( arguments, 0 ) );
    },
    
    // so it can be switched
    consoleLogFunction: function() {
      // allow for the console to not exist
      window.console && window.console.log && window.console.log.apply( window.console, Array.prototype.slice.call( arguments, 0 ) );
    },
    switchLogToConsole: function() {
      scenery.logFunction = scenery.consoleLogFunction;
    },
    
    stringLogFunction: function( message ) {
      scenery.logString += message.replace( /%c/g, '' ) + '\n';
    },
    switchLogToString: function() {
      scenery.logFunction = scenery.stringLogFunction;
    },
    
    enableLayerLogging: function() {
      window.sceneryLog = function( ob ) { scenery.logFunction( ob ); };
      
      var padding = '';
      window.sceneryLog.push = function() {
        padding += '| ';
      };
      window.sceneryLog.pop = function() {
        padding = padding.slice( 0, -2 );
      };
      
      var padStyle = 'color: #ddd;';
      
      // feature-specific debugging flags (so we don't log the entire world)
      // window.sceneryLog.dirty            = function( ob ) { scenery.logFunction( '%c' + padding + '%c[dirty] '       + ob, padStyle, 'color: #aaa;' ); };
      // window.sceneryLog.bounds           = function( ob ) { scenery.logFunction( '%c' + padding + '%c[bounds] '      + ob, padStyle, 'color: #aaa;' ); };
      // window.sceneryLog.hitTest          = function( ob ) { scenery.logFunction( '%c' + padding + '%c[hitTest] '     + ob, padStyle, 'color: #aaa;' ); };
      // window.sceneryLog.Cursor           = function( ob ) { scenery.logFunction( '%c' + padding + '%c[Cursor] '      + ob, padStyle, 'color: #000;' ); };
      
      // window.sceneryLog.transformSystem  = function( ob ) { scenery.logFunction( '%c' + padding + '%c[transform] '   + ob, padStyle, 'color: #606;' ); };
      window.sceneryLog.BackboneDrawable = function( ob ) { scenery.logFunction( '%c' + padding + '%c[Backbone] '    + ob, padStyle, 'color: #a00;' ); };
      window.sceneryLog.CanvasBlock      = function( ob ) { scenery.logFunction( '%c' + padding + '%c[Canvas] '      + ob, padStyle, 'color: #000;' ); };
      window.sceneryLog.Display          = function( ob ) { scenery.logFunction( '%c' + padding + '%c[Display] '     + ob, padStyle, 'color: #000;' ); };
      window.sceneryLog.Drawable         = function( ob ) { scenery.logFunction( '%c' + padding + '%c'               + ob, padStyle, 'color: #000;' ); };
      window.sceneryLog.FittedBlock      = function( ob ) { scenery.logFunction( '%c' + padding + '%c[FittedBlock] ' + ob, padStyle, 'color: #000;' ); };
      window.sceneryLog.Instance         = function( ob ) { scenery.logFunction( '%c' + padding + '%c[Instance] '    + ob, padStyle, 'color: #000;' ); };
      window.sceneryLog.SVGBlock         = function( ob ) { scenery.logFunction( '%c' + padding + '%c[SVG] '         + ob, padStyle, 'color: #000;' ); };
      window.sceneryLog.SVGGroup         = function( ob ) { scenery.logFunction( '%c' + padding + '%c[SVGGroup] '    + ob, padStyle, 'color: #000;' ); };
    },
  
    disableLayerLogging: function() {
      window.sceneryLog = null;
    },
    
    enableEventLogging: function() {
      window.sceneryEventLog = function( ob ) { scenery.logFunction( ob ); };
    },
  
    disableEventLogging: function() {
      window.sceneryEventLog = null;
    },
    
    enableAccessibilityLogging: function() {
      window.sceneryAccessibilityLog = function( ob ) { scenery.logFunction( ob ); };
    },
  
    disableAccessibilityLogging: function() {
      window.sceneryAccessibilityLog = null;
    },
    
    //OHTWO TODO: remove duplication between here and Renderer?
    bitmaskAll:            0xFFFFFFF, // 28 bits for now (don't go over 31 bits, or we'll see a 32-bit platform slowdown!)
    bitmaskNodeDefault:    0x00003FF,
    bitmaskPaintedDefault: 0x0000200, // bounds valid, no renderer set
    bitmaskRendererArea:   0x00000FF,
    
    // NOTE! If these are changed, please examine flags included in Renderer.js to make sure there are no conflicts (we use the same bitmask space)
    bitmaskSupportsCanvas: 0x0000001,
    bitmaskSupportsSVG:    0x0000002,
    bitmaskSupportsDOM:    0x0000004,
    bitmaskSupportsWebGL:  0x0000008,
    // 10, 20, 40, 80 reserved for future renderers
    bitmaskNotPainted:     0x0000100,
    bitmaskBoundsValid:    0x0000200  // i.e. painted area will not spill outside of bounds
    // TODO: what else would we need?
  };
  
  return scenery;
} );
