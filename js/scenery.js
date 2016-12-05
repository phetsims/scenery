// Copyright 2013-2016, University of Colorado Boulder

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

  var Namespace = require( 'PHET_CORE/Namespace' );
  var extend = require( 'PHET_CORE/extend' );

  // @public (scenery-internal)
  window.sceneryLog = null;
  window.sceneryAccessibilityLog = null;

  // Initialize object allocation tracking, if it hasn't been already.
  window.phetAllocation = require( 'PHET_CORE/phetAllocation' );

  var scratchCanvas = document.createElement( 'canvas' );
  var scratchContext = scratchCanvas.getContext( '2d' );

  var logPadding = '';

  var scenery = new Namespace( 'scenery' );

  // @public - A Canvas and 2D Canvas context used for convenience functions (think of it as having arbitrary state).
  scenery.register( 'scratchCanvas', scratchCanvas );
  scenery.register( 'scratchContext', scratchContext );

  // @public - SVG namespace, used for document.createElementNS( scenery.svgns, name );
  scenery.register( 'svgns', 'http://www.w3.org/2000/svg' );

  // @public - X-link namespace, used for SVG image URLs (xlink:href)
  scenery.register( 'xlinkns', 'http://www.w3.org/1999/xlink' );

  /*---------------------------------------------------------------------------*
   * Logging
   * TODO: Move this out of scenery.js if possible
   *---------------------------------------------------------------------------*/

  // @private - Scenery internal log function to be used to log to scenery.logString (does not include color/css)
  function stringLogFunction( message ) {
    scenery.logString += message.replace( /%c/g, '' ) + '\n';
  }

  // @private - Scenery internal log function to be used to log to the console.
  function consoleLogFunction() {
    // allow for the console to not exist
    window.console && window.console.log && window.console.log.apply( window.console, Array.prototype.slice.call( arguments, 0 ) );
  }

  // @private - List of Scenery's loggers, with their display name and (if using console) the display style.
  var logProperties = {
    dirty: { name: 'dirty', style: 'color: #aaa;' },
    bounds: { name: 'bounds', style: 'color: #aaa;' },
    hitTest: { name: 'hitTest', style: 'color: #aaa;' },
    hitTestInternal: { name: 'hitTestInternal', style: 'color: #aaa;' },
    PerfCritical: { name: 'Perf', style: 'color: #f00;' },
    PerfMajor: { name: 'Perf', style: 'color: #aa0;' },
    PerfMinor: { name: 'Perf', style: 'color: #088;' },
    PerfVerbose: { name: 'Perf', style: 'color: #888;' },
    Cursor: { name: 'Cursor', style: 'color: #000;' },
    Stitch: { name: 'Stitch', style: 'color: #000;' },
    StitchDrawables: { name: 'Stitch', style: 'color: #000;' },
    GreedyStitcher: { name: 'Greedy', style: 'color: #088;' },
    GreedyVerbose: { name: 'Greedy', style: 'color: #888;' },
    RelativeTransform: { name: 'RelativeTransform', style: 'color: #606;' },
    BackboneDrawable: { name: 'Backbone', style: 'color: #a00;' },
    CanvasBlock: { name: 'Canvas', style: 'color: #000;' },
    WebGLBlock: { name: 'WebGL', style: 'color: #000;' },
    Display: { name: 'Display', style: 'color: #000;' },
    DOMBlock: { name: 'DOM', style: 'color: #000;' },
    Drawable: { name: '', style: 'color: #000;' },
    FittedBlock: { name: 'FittedBlock', style: 'color: #000;' },
    Input: { name: 'Input', style: 'color: #000;' },
    InputEvent: { name: 'InputEvent', style: 'color: #000;' },
    Instance: { name: 'Instance', style: 'color: #000;' },
    InstanceTree: { name: 'InstanceTree', style: 'color: #000;' },
    ChangeInterval: { name: 'ChangeInterval', style: 'color: #0a0;' },
    SVGBlock: { name: 'SVG', style: 'color: #000;' },
    SVGGroup: { name: 'SVGGroup', style: 'color: #000;' },
    ImageSVGDrawable: { name: 'ImageSVGDrawable', style: 'color: #000;' },
    Paints: { name: 'Paints', style: 'color: #000;' },
    Accessibility: { name: 'Accessibility', style: 'color: #000;' },
    AccessibleInstance: { name: 'AccessibleInstance', style: 'color: #000;' },
    AlignBox: { name: 'AlignBox', style: 'color: #000;' },
    AlignGroup: { name: 'AlignGroup', style: 'color: #000;' }
  };

  // will be filled in by other modules
  extend( scenery, {
    // @public - Scenery log string (accumulated if switchLogToString() is used).
    logString: '',

    // @private - Scenery internal log function (switchable implementation, the main reference)
    logFunction: function() {
      // allow for the console to not exist
      window.console && window.console.log && window.console.log.apply( window.console, Array.prototype.slice.call( arguments, 0 ) );
    },

    // @public - Switches Scenery's logging to print to the developer console.
    switchLogToConsole: function() {
      scenery.logFunction = consoleLogFunction;
    },

    // @public - Switches Scenery's logging to append to scenery.logString
    switchLogToString: function() {
      window.console && window.console.log( 'switching to string log' );
      scenery.logFunction = stringLogFunction;
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
        assert && assert( logProperties[ name ],
          'Unknown logger: ' + name + ', please use periods (.) to separate different log names' );

        window.sceneryLog[ name ] = window.sceneryLog[ name ] || function( ob, styleOverride ) {
            var data = logProperties[ name ];

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
     * @param {Array.<string>} logNames - keys from logProperties
     */
    enableLogging: function( logNames ) {
      window.sceneryLog = function( ob ) { scenery.logFunction( ob ); };

      window.sceneryLog.push = function() {
        logPadding += '| ';
      };
      window.sceneryLog.pop = function() {
        logPadding = logPadding.slice( 0, -2 );
      };
      window.sceneryLog.getDepth = function() {
        return logPadding.length / 2;
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
    }
  } );

  return scenery;
} );
