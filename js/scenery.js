// Copyright 2013-2015, University of Colorado Boulder

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

  // @public (scenery-internal)
  window.sceneryLog = null;
  window.sceneryEventLog = null;
  window.sceneryAccessibilityLog = null;

  // Initialize object allocation tracking, if it hasn't been already.
  window.phetAllocation = require( 'PHET_CORE/phetAllocation' );

  var scratchCanvas = document.createElement( 'canvas' );
  var scratchContext = scratchCanvas.getContext( '2d' );

  var logPadding = '';

  // will be filled in by other modules
  var scenery = {
    // @public - A Canvas and 2D Canvas context used for convenience functions (think of it as having arbitrary state).
    scratchCanvas: scratchCanvas,
    scratchContext: scratchContext,

    // @public - SVG namespace, used for document.createElementNS( scenery.svgns, name );
    svgns: 'http://www.w3.org/2000/svg',

    // @public - X-link namespace, used for SVG image URLs (xlink:href)
    xlinkns: 'http://www.w3.org/1999/xlink',

    // @public - Scenery log string (accumulated if switchLogToString() is used).
    logString: '',

    // @private - Scenery internal log function (switchable implementation, the main reference)
    logFunction: function() {
      // allow for the console to not exist
      window.console && window.console.log && window.console.log.apply( window.console, Array.prototype.slice.call( arguments, 0 ) );
    },

    // @private - Scenery internal log function to be used to log to the console.
    consoleLogFunction: function() {
      // allow for the console to not exist
      window.console && window.console.log && window.console.log.apply( window.console, Array.prototype.slice.call( arguments, 0 ) );
    },

    // @public - Switches Scenery's logging to print to the developer console.
    switchLogToConsole: function() {
      scenery.logFunction = scenery.consoleLogFunction;
    },

    // @private - Scenery internal log function to be used to log to scenery.logString (does not include color/css)
    stringLogFunction: function( message ) {
      scenery.logString += message.replace( /%c/g, '' ) + '\n';
    },

    // @public - Switches Scenery's logging to append to scenery.logString
    switchLogToString: function() {
      window.console && window.console.log( 'switching to string log' );
      scenery.logFunction = scenery.stringLogFunction;
    },

    // @private - List of Scenery's loggers, with their display name and (if using console) the display style.
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
      RelativeTransform: { name: 'transform', style: 'color: #606;' },
      BackboneDrawable: { name: 'Backbone', style: 'color: #a00;' },
      CanvasBlock: { name: 'Canvas', style: 'color: #000;' },
      WebGLBlock: { name: 'WebGL', style: 'color: #000;' },
      Display: { name: 'Display', style: 'color: #000;' },
      DOMBlock: { name: 'DOM', style: 'color: #000;' },
      Drawable: { name: '', style: 'color: #000;' },
      FittedBlock: { name: 'FittedBlock', style: 'color: #000;' },
      Input: { name: 'Input', style: 'color: #000;' },
      Instance: { name: 'Instance', style: 'color: #000;' },
      InstanceTree: { name: 'InstanceTree', style: 'color: #000;' },
      ChangeInterval: { name: 'ChangeInterval', style: 'color: #0a0;' },
      SVGBlock: { name: 'SVG', style: 'color: #000;' },
      SVGGroup: { name: 'SVGGroup', style: 'color: #000;' },
      ImageSVGDrawable: { name: 'ImageSVGDrawable', style: 'color: #000;' },
      Paints: { name: 'Paints', style: 'color: #000;' },
      Accessibility: { name: 'Accessibility', style: 'color: #000;' }
    },

    // @public - Enables a specific single logger, OR a composite logger ('stitch'/'perf')
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
        assert && assert( scenery.logProperties[ name ],
          'Unknown logger: ' + name + ', please use periods (.) to separate different log names' );

        window.sceneryLog[ name ] = window.sceneryLog[ name ] || function( ob, styleOverride ) {
          var data = scenery.logProperties[ name ];

          var prefix = data.name ? '[' + data.name + '] ' : '';
          var padStyle = 'color: #ddd;';
          scenery.logFunction( '%c' + logPadding + '%c' + prefix + ob, padStyle, styleOverride ? styleOverride : data.style );
        };
      }
    },

    // @public - Disables a specific log. TODO: handle stitch and perf composite loggers
    disableIndividualLog: function( name ) {
      if ( name ) {
        delete window.sceneryLog[ name ];
      }
    },

    /**
     * Enables multiple loggers.
     * @public
     *
     * @param {Array.<string>} logNames - keys from scenery.logProperties
     */
    enableLogging: function( logNames ) {
      if ( !logNames ) {
        logNames = [ 'stitch' ];
      }

      window.sceneryLog = function( ob ) { scenery.logFunction( ob ); };

      window.sceneryLog.push = function() {
        logPadding += '| ';
      };
      window.sceneryLog.pop = function() {
        logPadding = logPadding.slice( 0, -2 );
      };

      for ( var i = 0; i < logNames.length; i++ ) {
        this.enableIndividualLog( logNames[ i ] );
      }
    },

    // @public - Disables Scenery logging
    disableLogging: function() {
      window.sceneryLog = null;
    },

    // @public (scenery-internal) - Whether performance logging is active (may actually reduce performance)
    isLoggingPerformance: function() {
      return window.sceneryLog.PerfCritical || window.sceneryLog.PerfMajor ||
             window.sceneryLog.PerfMinor || window.sceneryLog.PerfVerbose;
    },

    // @public @deprecated (scenery-internal) - Enables logging related to events
    enableEventLogging: function() {
      window.sceneryEventLog = function( ob ) { scenery.logFunction( ob ); };
    },

    // @public @deprecated (scenery-internal) - Disables logging related to events
    disableEventLogging: function() {
      window.sceneryEventLog = null;
    },

    // @public @deprecated (scenery-internal) - Enables logging related to accessibility
    enableAccessibilityLogging: function() {
      window.sceneryAccessibilityLog = function( ob ) { scenery.logFunction( ob ); };
    },

    // @public @deprecated (scenery-internal) - Disables logging related to accessibility
    disableAccessibilityLogging: function() {
      window.sceneryAccessibilityLog = null;
    }
  };

  // store a reference on the PhET namespace if it exists
  if ( window.phet ) {
    window.phet.scenery = scenery;
  }

  return scenery;
} );
