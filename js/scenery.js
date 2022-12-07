// Copyright 2013-2022, University of Colorado Boulder

/**
 * The main 'scenery' namespace object for the exported (built) API. Used internally in some places where there are
 * potential circular dependencies.
 *
 * The returned scenery object namespace may be incomplete if not all modules are listed as
 * dependencies. Please use the 'main' module for that purpose if all of Scenery is desired.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 *
 * TODO: When converting to TypeScript, please see ts-expect-error in SimDisplay
 */

import extend from '../../phet-core/js/extend.js';
import Namespace from '../../phet-core/js/Namespace.js';

// @public (scenery-internal)
window.sceneryLog = null;

const scratchCanvas = document.createElement( 'canvas' );
const scratchContext = scratchCanvas.getContext( '2d' );

let logPadding = '';

const scenery = new Namespace( 'scenery' );

// @public - A Canvas and 2D Canvas context used for convenience functions (think of it as having arbitrary state).
scenery.register( 'scratchCanvas', scratchCanvas );
scenery.register( 'scratchContext', scratchContext );

/*---------------------------------------------------------------------------*
 * Logging
 * TODO: Move this out of scenery.js if possible
 *---------------------------------------------------------------------------*/

// @private - Scenery internal log function to be used to log to scenery.logString (does not include color/css)
function stringLogFunction( message ) {
  scenery.logString += `${message.replace( /%c/g, '' )}\n`;
}

// @private - Scenery internal log function to be used to log to the console.
function consoleLogFunction( ...args ) {
  // allow for the console to not exist
  window.console && window.console.log && window.console.log( ...Array.prototype.slice.call( args, 0 ) );
}

// @private - List of Scenery's loggers, with their display name and (if using console) the display style.
const logProperties = {
  dirty: { name: 'dirty', style: 'color: #888;' },
  bounds: { name: 'bounds', style: 'color: #888;' },
  hitTest: { name: 'hitTest', style: 'color: #888;' },
  hitTestInternal: { name: 'hitTestInternal', style: 'color: #888;' },
  PerfCritical: { name: 'Perf', style: 'color: #f00;' },
  PerfMajor: { name: 'Perf', style: 'color: #aa0;' },
  PerfMinor: { name: 'Perf', style: 'color: #088;' },
  PerfVerbose: { name: 'Perf', style: 'color: #888;' },
  Cursor: { name: 'Cursor', style: '' },
  Stitch: { name: 'Stitch', style: '' },
  StitchDrawables: { name: 'Stitch', style: '' },
  GreedyStitcher: { name: 'Greedy', style: 'color: #088;' },
  GreedyVerbose: { name: 'Greedy', style: 'color: #888;' },
  RelativeTransform: { name: 'RelativeTransform', style: 'color: #606;' },
  BackboneDrawable: { name: 'Backbone', style: 'color: #a00;' },
  CanvasBlock: { name: 'Canvas', style: '' },
  WebGLBlock: { name: 'WebGL', style: '' },
  Display: { name: 'Display', style: '' },
  DOMBlock: { name: 'DOM', style: '' },
  Drawable: { name: '', style: '' },
  FittedBlock: { name: 'FittedBlock', style: '' },
  Instance: { name: 'Instance', style: '' },
  InstanceTree: { name: 'InstanceTree', style: '' },
  ChangeInterval: { name: 'ChangeInterval', style: 'color: #0a0;' },
  SVGBlock: { name: 'SVG', style: '' },
  SVGGroup: { name: 'SVGGroup', style: '' },
  ImageSVGDrawable: { name: 'ImageSVGDrawable', style: '' },
  Paints: { name: 'Paints', style: '' },
  Filters: { name: 'Filters', style: '' },
  AlignBox: { name: 'AlignBox', style: '' },
  AlignGroup: { name: 'AlignGroup', style: '' },
  RichText: { name: 'RichText', style: '' },

  Sim: { name: 'Sim', style: '' },

  // Accessibility-related
  ParallelDOM: { name: 'ParallelDOM', style: '' },
  PDOMInstance: { name: 'PDOMInstance', style: '' },
  PDOMTree: { name: 'PDOMTree', style: '' },
  PDOMDisplaysInfo: { name: 'PDOMDisplaysInfo', style: '' },
  KeyboardFuzzer: { name: 'KeyboardFuzzer', style: '' },

  // Input-related
  InputListener: { name: 'InputListener', style: '' },
  InputEvent: { name: 'InputEvent', style: '' },
  OnInput: { name: 'OnInput', style: '' },
  Pointer: { name: 'Pointer', style: '' },
  Input: { name: 'Input', style: '' }, // When "logical" input functions are called, and related tasks
  EventDispatch: { name: 'EventDispatch', style: '' }, // When an event is dispatched, and when listeners are triggered
  EventPath: { name: 'EventPath', style: '' } // User-readable form for whenever an event is dispatched
};

// will be filled in by other modules
extend( scenery, {
  // @public - Scenery log string (accumulated if switchLogToString() is used).
  logString: '',

  // @private - Scenery internal log function (switchable implementation, the main reference)
  logFunction: function( ...args ) {
    // allow for the console to not exist
    window.console && window.console.log && window.console.log( ...Array.prototype.slice.call( args, 0 ) );
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

    if ( name === 'input' ) {
      this.enableIndividualLog( 'InputListener' );
      this.enableIndividualLog( 'InputEvent' );
      this.enableIndividualLog( 'OnInput' );
      this.enableIndividualLog( 'Pointer' );
      this.enableIndividualLog( 'Input' );
      this.enableIndividualLog( 'EventDispatch' );
      this.enableIndividualLog( 'EventPath' );
      return;
    }
    if ( name === 'a11y' || name === 'pdom' ) {
      this.enableIndividualLog( 'ParallelDOM' );
      this.enableIndividualLog( 'PDOMInstance' );
      this.enableIndividualLog( 'PDOMTree' );
      this.enableIndividualLog( 'PDOMDisplaysInfo' );
      return;
    }

    if ( name ) {
      assert && assert( logProperties[ name ],
        `Unknown logger: ${name}, please use periods (.) to separate different log names` );

      window.sceneryLog[ name ] = window.sceneryLog[ name ] || function( ob, styleOverride ) {
        const data = logProperties[ name ];

        const prefix = data.name ? `[${data.name}] ` : '';
        const padStyle = 'color: #ddd;';
        scenery.logFunction( `%c${logPadding}%c${prefix}${ob}`, padStyle, styleOverride ? styleOverride : data.style );
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

    for ( let i = 0; i < logNames.length; i++ ) {
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

export default scenery;