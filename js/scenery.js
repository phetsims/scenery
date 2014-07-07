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

  var logPadding = '';

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

    logProperties: {
      dirty: { name: 'dirty', style: 'color: #aaa;' },
      bounds: { name: 'bounds', style: 'color: #aaa;' },
      hitTest: { name: 'hitTest', style: 'color: #aaa;' },
      PerfCritical: { name: 'Perf', style: 'color: #f00;' },
      PerfMajor: { name: 'Perf', style: 'color: #aa0;' },
      PerfMinor: { name: 'Perf', style: 'color: #088;' },
      PerfVerbose: { name: 'Perf', style: 'color: #888;' },
      Cursor: { name: 'Cursor', style: 'color: #000;' },
      Stitch: { name: 'Stitch', style: 'color: #000;' },
      StitchDrawables: { name: 'Stitch', style: 'color: #000;' },
      GreedyStitcher: { name: 'Greedy', style: 'color: #088;' },
      GreedyVerbose: { name: 'Greedy', style: 'color: #888;' },
      transformSystem: { name: 'transform', style: 'color: #606;' },
      BackboneDrawable: { name: 'Backbone', style: 'color: #a00;' },
      CanvasBlock: { name: 'Canvas', style: 'color: #000;' },
      Display: { name: 'Display', style: 'color: #000;' },
      DOMBlock: { name: 'DOM', style: 'color: #000;' },
      Drawable: { name: '', style: 'color: #000;' },
      FittedBlock: { name: 'FittedBlock', style: 'color: #000;' },
      Input: { name: 'Input', style: 'color: #000;' },
      Instance: { name: 'Instance', style: 'color: #000;' },
      SVGBlock: { name: 'SVG', style: 'color: #000;' },
      SVGGroup: { name: 'SVGGroup', style: 'color: #000;' },
      RenderState: { name: 'RenderState', style: 'color: #000;' }
    },
    enableIndividualLog: function( name ) {
      if ( name === 'stitch' ) {
        this.enableIndividualLog( 'Stitch' );
        this.enableIndividualLog( 'StitchDrawables' );
        this.enableIndividualLog( 'GreedyStitcher' );
        this.enableIndividualLog( 'GreedyVerbose' );
        return;
      }

      if ( name === 'perf' ) {
        this.enableIndividualLog( 'PerfCritical' );
        this.enableIndividualLog( 'PerfMajor' );
        this.enableIndividualLog( 'PerfMinor' );
        this.enableIndividualLog( 'PerfVerbose' );
        return;
      }

      if ( name ) {
        assert && assert( scenery.logProperties[name], 'Unknown logger: ' + name );

        window.sceneryLog[name] = window.sceneryLog[name] || function( ob, styleOverride ) {
          var data = scenery.logProperties[name];

          var prefix = data.name ? '[' + data.name + '] ' : '';
          var padStyle = 'color: #ddd;';
          scenery.logFunction( '%c' + logPadding + '%c' + prefix + ob, padStyle, styleOverride ? styleOverride : data.style );
        };
      }
    },
    disableIndividualLog: function( name ) {
      if ( name ) {
        delete window.sceneryLog[name];
      }
    },
    enableLogging: function( logNames ) {
      if ( !logNames ) {
        logNames = ['stitch'];
      }

      window.sceneryLog = function( ob ) { scenery.logFunction( ob ); };

      window.sceneryLog.push = function() {
        logPadding += '| ';
      };
      window.sceneryLog.pop = function() {
        logPadding = logPadding.slice( 0, -2 );
      };

      for ( var i = 0; i < logNames.length; i++ ) {
        this.enableIndividualLog( logNames[i] );
      }
    },

    disableLogging: function() {
      window.sceneryLog = null;
    },

    isLoggingPerformance: function() {
      return window.sceneryLog.PerfCritical || window.sceneryLog.PerfMajor ||
             window.sceneryLog.PerfMinor || window.sceneryLog.PerfVerbose;
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
    bitmaskAll: 0xFFFFFFF, // 28 bits for now (don't go over 31 bits, or we'll see a 32-bit platform slowdown!)
    bitmaskNodeDefault: 0x00003FF,
    bitmaskPaintedDefault: 0x0000200, // bounds valid, no renderer set
    bitmaskRendererArea: 0x00000FF,

    // NOTE! If these are changed, please examine flags included in Renderer.js to make sure there are no conflicts (we use the same bitmask space)
    bitmaskSupportsCanvas: 0x0000001,
    bitmaskSupportsSVG: 0x0000002,
    bitmaskSupportsDOM: 0x0000004,
    bitmaskSupportsWebGL: 0x0000008,
    // 10, 20, 40, 80 reserved for future renderers
    bitmaskNotPainted: 0x0000100,
    bitmaskBoundsValid: 0x0000200  // i.e. painted area will not spill outside of bounds
    // TODO: what else would we need?
  };

  return scenery;
} );
