// Copyright 2002-2014, University of Colorado

/**
 * A Path draws a Shape with a specific type of fill and stroke.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Shape = require( 'KITE/Shape' );

  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );
  require( 'SCENERY/display/Renderer' );
  var Fillable = require( 'SCENERY/nodes/Fillable' );
  var Strokable = require( 'SCENERY/nodes/Strokable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepSVGPathElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  scenery.Path = function Path( shape, options ) {
    // TODO: consider directly passing in a shape object (or at least handling that case)
    // NOTE: _shape can be lazily constructed, in the case of types like Rectangle where they have their own drawing code
    this._shape = null;
    this._strokedShape = null; // a stroked copy of the shape, lazily computed

    // ensure we have a parameter object
    options = options || {};

    this.initializeFillable();
    this.initializeStrokable();

    Node.call( this );
    this.invalidateSupportedRenderers();
    this.setShape( shape );
    this.mutate( options );
  };
  var Path = scenery.Path;

  inherit( Node, Path, {
    // allow more specific path types (Rectangle, Line) to override what restrictions we have
    getPathRendererBitmask: function() {
      return scenery.bitmaskBoundsValid | scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG;
    },

    invalidateSupportedRenderers: function() {
      this.setRendererBitmask( this.getFillRendererBitmask() & this.getStrokeRendererBitmask() & this.getPathRendererBitmask() );
    },

    // sets the shape drawn, or null to remove the shape
    setShape: function( shape ) {
      if ( this._shape !== shape ) {
        if ( typeof shape === 'string' ) {
          // be content with setShape always invalidating the shape?
          shape = new Shape( shape );
        }
        this._shape = shape;
        this.invalidateShape();

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[i].markDirtyShape();
        }
      }
      return this;
    },

    getShape: function() {
      return this._shape;
    },

    getStrokedShape: function() {
      if ( !this._strokedShape ) {
        this._strokedShape = this.getShape().getStrokedShape( this._lineDrawingStyles );
      }
      return this._strokedShape;
    },

    invalidateShape: function() {
      this._strokedShape = null;

      if ( this.hasShape() ) {
        this.invalidateSelf( this.computeShapeBounds() );
      }
    },

    // separated out, so that we can override this with a faster version in subtypes. includes the Stroke, if any
    computeShapeBounds: function() {
      return this._stroke ? this.getStrokedShape().bounds : this.getShape().bounds;
    },

    // hook stroke mixin changes to invalidation
    invalidateStroke: function() {
      this.invalidateShape();
    },

    hasShape: function() {
      return this._shape;
    },

    canvasPaintSelf: function( wrapper ) {
      Path.PathCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    createSVGDrawable: function( renderer, instance ) {
      return Path.PathSVGDrawable.createFromPool( renderer, instance );
    },

    createCanvasDrawable: function( renderer, instance ) {
      return Path.PathCanvasDrawable.createFromPool( renderer, instance );
    },

    isPainted: function() {
      return true;
    },

    // override for computation of whether a point is inside the self content
    // point is considered to be in the local coordinate frame
    containsPointSelf: function( point ) {
      var result = false;
      if ( !this.hasShape() ) {
        return result;
      }

      // if this node is fillPickable, we will return true if the point is inside our fill area
      if ( this._fillPickable ) {
        result = this.getShape().containsPoint( point );
      }

      // also include the stroked region in the hit area if strokePickable
      if ( !result && this._strokePickable ) {
        result = this.getStrokedShape().containsPoint( point );
      }
      return result;
    },

    // whether this node's self intersects the specified bounds, in the local coordinate frame
    intersectsBoundsSelf: function( bounds ) {
      // TODO: should a shape's stroke be included?
      return this.hasShape() ? this._shape.intersectsBounds( bounds ) : false;
    },

    set shape( value ) { this.setShape( value ); },
    get shape() { return this.getShape(); },

    getDebugHTMLExtras: function() {
      return this._shape ? ' (<span style="color: #88f" onclick="window.open( \'data:text/plain;charset=utf-8,\' + encodeURIComponent( \'' + this._shape.getSVGPath() + '\' ) );">path</span>)' : '';
    },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Path( ' + ( this._shape ? this._shape.toString() : this._shape ) + ', {' + propLines + '} )';
    },

    getPropString: function( spaces, includeChildren ) {
      var result = Node.prototype.getPropString.call( this, spaces, includeChildren );
      result = this.appendFillablePropString( spaces, result );
      result = this.appendStrokablePropString( spaces, result );
      return result;
    }
  } );

  Path.prototype._mutatorKeys = [ 'shape' ].concat( Node.prototype._mutatorKeys );

  // mix in fill/stroke handling code. for now, this is done after 'shape' is added to the mutatorKeys so that stroke parameters
  // get set first
  /* jshint -W064 */
  Fillable( Path );
  Strokable( Path );

  /*---------------------------------------------------------------------------*
  * Rendering State mixin (DOM/SVG)
  *----------------------------------------------------------------------------*/

  var PathRenderState = Path.PathRenderState = function( drawableType ) {
    var proto = drawableType.prototype;

    // initializes, and resets (so we can support pooled states)
    proto.initializeState = function() {
      this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
      this.dirtyShape = true;

      // adds fill/stroke-specific flags and state
      this.initializeFillableState();
      this.initializeStrokableState();

      return this; // allow for chaining
    };

    // catch-all dirty, if anything that isn't a transform is marked as dirty
    proto.markPaintDirty = function() {
      this.paintDirty = true;
      this.markDirty();
    };

    proto.markDirtyShape = function() {
      this.dirtyShape = true;
      this.markPaintDirty();
    };

    proto.setToCleanState = function() {
      this.paintDirty = false;
      this.dirtyShape = false;

      this.cleanFillableState();
      this.cleanStrokableState();
    };

    /* jshint -W064 */
    Fillable.FillableState( drawableType );
    /* jshint -W064 */
    Strokable.StrokableState( drawableType );
  };

  /*---------------------------------------------------------------------------*
  * SVG Rendering
  *----------------------------------------------------------------------------*/

  Path.PathSVGDrawable = SVGSelfDrawable.createDrawable( {
    type: function PathSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    stateType: PathRenderState,
    initialize: function( renderer, instance ) {
      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'path' );
      }
    },
    updateSVG: function( node, path ) {
      assert && assert( !node.requiresSVGBoundsWorkaround(), 'No workaround for https://github.com/phetsims/scenery/issues/196 is provided at this time, please add an epsilon' );

      if ( this.dirtyShape ) {
        var svgPath = node.hasShape() ? node._shape.getSVGPath() : '';

        // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
        // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
        // the attribute can cause this bug
        if ( !svgPath ) { svgPath = 'M0 0'; }

        // only set the SVG path if it's not the empty string
        path.setAttribute( 'd', svgPath );
      }

      this.updateFillStrokeStyle( path );
    },
    usesFill: true,
    usesStroke: true,
    keepElements: keepSVGPathElements
  } );

  /*---------------------------------------------------------------------------*
  * Canvas rendering
  *----------------------------------------------------------------------------*/

  Path.PathCanvasDrawable = CanvasSelfDrawable.createDrawable( {
    type: function PathCanvasDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    paintCanvas: function paintCanvasPath( wrapper, node ) {
      var context = wrapper.context;

      if ( node.hasShape() ) {
        // TODO: fill/stroke delay optimizations?
        context.beginPath();
        node._shape.writeToContext( context );

        if ( node._fill ) {
          node.beforeCanvasFill( wrapper ); // defined in Fillable
          context.fill();
          node.afterCanvasFill( wrapper ); // defined in Fillable
        }
        if ( node._stroke ) {
          node.beforeCanvasStroke( wrapper ); // defined in Strokable
          context.stroke();
          node.afterCanvasStroke( wrapper ); // defined in Strokable
        }
      }
    },
    usesFill: true,
    usesStroke: true,
    dirtyMethods: ['markDirtyShape']
  } );

  return Path;
} );


