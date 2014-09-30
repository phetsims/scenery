// Copyright 2002-2014, University of Colorado Boulder

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

    bitmaskAll: 0xFFFFFFF, // 28 bits for now (don't go over 31 bits, or we'll see a 32-bit platform slowdown!)
    bitmaskNodeDefault: 0x00001FF,
    bitmaskPaintedDefault: 0x0000000,

    bitmaskSupportsCanvas: 0x0000001,
    bitmaskSupportsSVG: 0x0000002,
    bitmaskSupportsDOM: 0x0000004,
    bitmaskSupportsWebGL: 0x0000008,
    // 10, 20, 40, 80 reserved for future renderers
    bitmaskNotPainted: 0x0000100
    // TODO: what else would we need?
  };
} );
