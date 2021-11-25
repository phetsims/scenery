// Copyright 2013-2021, University of Colorado Boulder

/**
 * The main 'scenery' namespace object for the exported (built) API. Used internally in some places where there are
 * potential circular dependencies.
 *
 * The returned scenery object namespace may be incomplete if not all modules are listed as
 * dependencies. Please use the 'main' module for that purpose if all of Scenery is desired.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import axon from '../../axon/js/axon.js';
import dot from '../../dot/js/dot.js';
import kite from '../../kite/js/kite.js';
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
  },

  // @public
  copy: function( value ) {
    return scenery.deserialize( scenery.serialize( value ) );
  },

  // @public
  deserialize: function( value ) {
    const nodeTypes = [
      'Node', 'Path', 'Circle', 'Line', 'Rectangle', 'Text', 'Image', 'CanvasNode', 'WebGLNode', 'DOM'
    ];

    if ( value.type === 'Vector2' ) {
      return new dot.Vector2( value.x, value.y );
    }
    if ( value.type === 'Matrix3' ) {
      return new dot.Matrix3().rowMajor( value.m00, value.m01, value.m02,
        value.m10, value.m11, value.m12,
        value.m20, value.m21, value.m22 );
    }
    else if ( value.type === 'Bounds2' ) {
      return new dot.Bounds2( value.minX, value.minY, value.maxX, value.maxY );
    }
    else if ( value.type === 'Shape' ) {
      return new kite.Shape( value.path );
    }
    else if ( value.type === 'Array' ) {
      return value.value.map( scenery.deserialize );
    }
    else if ( value.type === 'Color' ) {
      return new scenery.Color( value.red, value.green, value.blue, value.alpha );
    }
    else if ( value.type === 'Property' ) {
      return new axon.Property( scenery.deserialize( value.value ) );
    }
    else if ( value.type === 'Pattern' || value.type === 'LinearGradient' || value.type === 'RadialGradient' ) {
      let paint;

      if ( value.type === 'Pattern' ) {
        const img = new window.Image();
        img.src = value.url;
        paint = new scenery.Pattern( img );
      }
      // Gradients
      else {
        const start = scenery.deserialize( value.start );
        const end = scenery.deserialize( value.end );
        if ( value.type === 'LinearGradient' ) {
          paint = new scenery.LinearGradient( start.x, start.y, end.x, end.y );
        }
        else if ( value.type === 'RadialGradient' ) {
          paint = new scenery.RadialGradient( start.x, start.y, value.startRadius, end.x, end.y, value.endRadius );
        }

        value.stops.forEach( stop => {
          paint.addColorStop( stop.ratio, scenery.deserialize( stop.stop ) );
        } );
      }

      if ( value.transformMatrix ) {
        paint.setTransformMatrix( scenery.deserialize( value.transformMatrix ) );
      }

      return paint;
    }
    else if ( _.includes( nodeTypes, value.type ) ) {
      let node;

      const setup = value.setup;

      if ( value.type === 'Node' ) {
        node = new scenery.Node();
      }
      else if ( value.type === 'Path' ) {
        node = new scenery.Path( scenery.deserialize( setup.path ) );
      }
      else if ( value.type === 'Circle' ) {
        node = new scenery.Circle( {} );
      }
      else if ( value.type === 'Line' ) {
        node = new scenery.Line( {} );
      }
      else if ( value.type === 'Rectangle' ) {
        node = new scenery.Rectangle( {} );
      }
      else if ( value.type === 'Text' ) {
        node = new scenery.Text( '' );
      }
      else if ( value.type === 'Image' ) {
        if ( setup.imageType === 'image' || setup.imageType === 'canvas' ) {
          node = new scenery.Image( setup.src );
          if ( setup.generateMipmaps ) {
            node.mipmaps = true;
          }
        }
        else if ( setup.imageType === 'mipmapData' ) {
          const mipmapData = setup.mipmapData.map( level => {
            const result = {
              width: level.width,
              height: level.height,
              url: level.url
            };
            result.img = new window.Image();
            result.img.src = level.url;
            result.canvas = document.createElement( 'canvas' );
            result.canvas.width = level.width;
            result.canvas.height = level.height;
            const context = result.canvas.getContext( '2d' );
            // delayed loading like in mipmap plugin
            result.updateCanvas = function() {
              if ( result.img.complete && ( typeof result.img.naturalWidth === 'undefined' || result.img.naturalWidth > 0 ) ) {
                context.drawImage( result.img, 0, 0 );
                delete result.updateCanvas;
              }
            };
            return result;
          } );
          node = new scenery.Image( mipmapData );
        }
        node.initialWidth = setup.width;
        node.initialHeight = setup.height;
      }
      else if ( value.type === 'CanvasNode' || value.type === 'WebGLNode' ) {
        // TODO: Record Canvas/WebGL calls? (conditionals would be harder!)
        node = new scenery.Node( {
          children: [
            new scenery.Image( setup.url, {
              translation: scenery.deserialize( setup.offset ),
              scale: 1 / setup.scale
            } )
          ]
        } );
      }
      else if ( value.type === 'DOM' ) {
        const div = document.createElement( 'div' );
        div.innerHTML = value.element;
        const element = div.childNodes[ 0 ];
        div.removeChild( element );

        if ( value.dataURL ) {
          const img = new window.Image();
          img.onload = () => {
            const context = element.getContext( '2d' );
            context.drawImage( img, 0, 0 );
          };
          img.src = value.dataURL;
        }

        node = new scenery.DOM( element );
      }

      if ( setup.clipArea ) {
        node.clipArea = scenery.deserialize( setup.clipArea );
      }
      if ( setup.mouseArea ) {
        node.mouseArea = scenery.deserialize( setup.mouseArea );
      }
      if ( setup.touchArea ) {
        node.touchArea = scenery.deserialize( setup.touchArea );
      }
      if ( setup.matrix ) {
        node.matrix = scenery.deserialize( setup.matrix );
      }
      if ( setup.localBounds ) {
        node.localBounds = scenery.deserialize( setup.localBounds );
      }

      // Paintable, if they exist
      if ( setup.fill ) {
        node.fill = scenery.deserialize( setup.fill );
      }
      if ( setup.stroke ) {
        node.stroke = scenery.deserialize( setup.stroke );
      }
      if ( setup.lineDash ) {
        node.lineDash = scenery.deserialize( setup.lineDash );
      }

      node.mutate( value.options );

      node._serialization = value;

      return node;
    }
    else if ( value.type === 'Subtree' ) {
      const nodeMap = {};
      const nodes = value.nodes.map( scenery.deserialize );

      // Index them
      nodes.forEach( node => {
        nodeMap[ node._serialization.id ] = node;
      } );

      // Connect children
      nodes.forEach( node => {
        node._serialization.setup.children.forEach( childId => {
          node.addChild( nodeMap[ childId ] );
        } );
      } );

      // The root should be the first one
      return nodeMap[ value.rootNodeId ];
    }
    else if ( value.type === 'value' ) {
      return value.value;
    }
    else {
      return null;
    }
  }
} );

export default scenery;