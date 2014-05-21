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
  
  window.sceneryLayerLog = null;
  window.sceneryEventLog = null;
  window.sceneryAccessibilityLog = null;
  
  // object allocation tracking
  window.phetAllocation = require( 'PHET_CORE/phetAllocation' );
  
  var scratchCanvas = document.createElement( 'canvas' );
  var scratchContext = scratchCanvas.getContext( '2d' );
  
  // will be filled in by other modules
  return {
    assert: assert,
    
    scratchCanvas: scratchCanvas,   // a canvas used for convenience functions (think of it as having arbitrary state)
    scratchContext: scratchContext, // a context used for convenience functions (think of it as having arbitrary state)
    
    svgns: 'http://www.w3.org/2000/svg',     // svg namespace
    xlinkns: 'http://www.w3.org/1999/xlink', // x-link namespace
    
    enableLayerLogging: function() {
      window.sceneryLayerLog = function( ob ) { console.log( ob ); };
      
      // feature-specific debugging flags (so we don't log the entire world)
      
      // window.sceneryLayerLog.dirty = function( ob ) { console.log( '[dirty] ' + ob ); };
      // window.sceneryLayerLog.BackboneDrawable = function( ob ) { console.log( '[Backbone] ' + ob ); };
      // window.sceneryLayerLog.FittedBlock = function( ob ) { console.log( '[FittedBlock] ' + ob ); };
      // window.sceneryLayerLog.Instance = function( ob ) { console.log( '[Instance] ' + ob ); };
      // window.sceneryLayerLog.SVGBlock = function( ob ) { console.log( '[SVG] ' + ob ); };
      // window.sceneryLayerLog.SVGGroup = function( ob ) { console.log( '[SVGGroup] ' + ob ); };
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
} );
