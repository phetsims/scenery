// Copyright 2002-2014, University of Colorado Boulder

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
  var Paintable = require( 'SCENERY/nodes/Paintable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );
  var PixiSelfDrawable = require( 'SCENERY/display/PixiSelfDrawable' );
  var SquareUnstrokedRectangle = require( 'SCENERY/display/webgl/SquareUnstrokedRectangle' );
  var Color = require( 'SCENERY/util/Color' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepSVGPathElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  scenery.Path = function Path( shape, options ) {
    // TODO: consider directly passing in a shape object (or at least handling that case)
    // NOTE: _shape can be lazily constructed, in the case of types like Rectangle where they have their own drawing code
    this._shape = null;
    this._strokedShape = null; // a stroked copy of the shape, lazily computed

    // ensure we have a parameter object
    options = options || {};

    this.initializePaintable();

    Node.call( this );
    this.invalidateSupportedRenderers();
    this.setShape( shape );
    this.mutate( options );
  };
  var Path = scenery.Path;

  inherit( Node, Path, {
    // allow more specific path types (Rectangle, Line) to override what restrictions we have
    getPathRendererBitmask: function() {
      return scenery.bitmaskBoundsValid | scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG | scenery.bitmaskSupportsPixi;
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
          this._drawables[ i ].markDirtyShape();
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

    // @override
    getTransformedSelfBounds: function( matrix ) {
      return ( this._stroke ? this.getStrokedShape() : this.getShape() ).getBoundsWithTransform( matrix );
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

    createWebGLDrawable: function( renderer, instance ) {
      return Path.PathWebGLDrawable.createFromPool( renderer, instance );
    },

    createPixiDrawable: function( renderer, instance ) {
      return Path.PathPixiDrawable.createFromPool( renderer, instance );
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

    // if we have to apply a transform workaround for https://github.com/phetsims/scenery/issues/196 (only when we have a pattern or gradient)
    requiresSVGBoundsWorkaround: function() {
      if ( !this._stroke || !this._stroke.getSVGDefinition || !this.hasShape() ) {
        return false;
      }

      var bounds = this.computeShapeBounds( false ); // without stroke
      return bounds.x * bounds.y === 0; // at least one of them was zero, so the bounding box has no area
    },

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
  Paintable.mixin( Path );

  /*---------------------------------------------------------------------------*
   * Rendering State mixin (DOM/SVG)
   *----------------------------------------------------------------------------*/

  Path.PathStatefulDrawable = {
    mixin: function( drawableType ) {
      var proto = drawableType.prototype;

      // initializes, and resets (so we can support pooled states)
      proto.initializeState = function() {
        this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
        this.dirtyShape = true;

        // adds fill/stroke-specific flags and state
        this.initializePaintableState();

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

        this.cleanPaintableState();
      };

      Paintable.PaintableStatefulDrawable.mixin( drawableType );
    }
  };

  /*---------------------------------------------------------------------------*
   * SVG Rendering
   *----------------------------------------------------------------------------*/

  Path.PathSVGDrawable = function PathSVGDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( SVGSelfDrawable, Path.PathSVGDrawable, {
    initialize: function( renderer, instance ) {
      this.initializeSVGSelfDrawable( renderer, instance, true, keepSVGPathElements ); // usesPaint: true

      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'path' );
      }

      return this;
    },

    updateSVGSelf: function() {
      assert && assert( !this.node.requiresSVGBoundsWorkaround(),
        'No workaround for https://github.com/phetsims/scenery/issues/196 is provided at this time, please add an epsilon' );

      var path = this.svgElement;
      if ( this.dirtyShape ) {
        var svgPath = this.node.hasShape() ? this.node._shape.getSVGPath() : '';

        // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
        // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
        // the attribute can cause this bug
        if ( !svgPath ) { svgPath = 'M0 0'; }

        // only set the SVG path if it's not the empty string
        path.setAttribute( 'd', svgPath );
      }

      this.updateFillStrokeStyle( path );
    }
  } );
  Path.PathStatefulDrawable.mixin( Path.PathSVGDrawable );
  SelfDrawable.Poolable.mixin( Path.PathSVGDrawable );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  Path.PathCanvasDrawable = function PathCanvasDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( CanvasSelfDrawable, Path.PathCanvasDrawable, {
    initialize: function( renderer, instance ) {
      return this.initializeCanvasSelfDrawable( renderer, instance );
    },

    paintCanvas: function( wrapper, node ) {
      var context = wrapper.context;

      if ( node.hasShape() ) {
        // TODO: fill/stroke delay optimizations?
        context.beginPath();
        node._shape.writeToContext( context );

        if ( node._fill ) {
          node.beforeCanvasFill( wrapper ); // defined in Paintable
          context.fill();
          node.afterCanvasFill( wrapper ); // defined in Paintable
        }
        if ( node._stroke ) {
          node.beforeCanvasStroke( wrapper ); // defined in Paintable
          context.stroke();
          node.afterCanvasStroke( wrapper ); // defined in Paintable
        }
      }
    },

    // stateless dirty functions
    markDirtyShape: function() { this.markPaintDirty(); }
  } );
  Paintable.PaintableStatelessDrawable.mixin( Path.PathCanvasDrawable );
  SelfDrawable.Poolable.mixin( Path.PathCanvasDrawable );

  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  Path.PathWebGLDrawable = inherit( WebGLSelfDrawable, function PathWebGLDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // called either from the constructor or from pooling
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );
    },

    initializeContext: function( webglBlock ) {
      this.webglBlock = webglBlock;
      this.rectangleHandle = new SquareUnstrokedRectangle( webglBlock.webGLRenderer.colorTriangleRenderer, this.node, 0.5 );

      // cleanup old vertexBuffer, if applicable
      this.disposeWebGLBuffers();

      this.initializePaintableState();
      this.updatePath();

      //TODO: Update the state in the buffer arrays
    },

    //Nothing necessary since everything currently handled in the uModelViewMatrix below
    //However, we may switch to dynamic draw, and handle the matrix change only where necessary in the future?
    updatePath: function() {

      // TODO: a way to update the ColorTriangleBufferData.

      // TODO: move to PaintableWebGLState???
      if ( this.dirtyFill ) {
        this.color = Color.toColor( this.node._fill || 'blue' );
        this.cleanPaintableState();
      }
    },

    render: function( shaderProgram ) {
      // This is handled by the ColorTriangleRenderer
    },

    dispose: function() {
      this.disposeWebGLBuffers();

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    },

    disposeWebGLBuffers: function() {
      this.webglBlock.webGLRenderer.colorTriangleRenderer.colorTriangleBufferData.dispose( this.rectangleHandle );
    },

    markDirtyShape: function() {
      this.markDirty();
    },

    // general flag set on the state, which we forward directly to the drawable's paint flag
    markPaintDirty: function() {
      this.markDirty();
    },

    //TODO: Make sure all of the dirty flags make sense here.  Should we be using fillDirty, paintDirty, dirty, etc?
    update: function() {
      if ( this.dirty ) {
        this.updatePath();
        this.dirty = false;
      }
    }
  } );
  // include stubs (stateless) for marking dirty stroke and fill (if necessary). we only want one dirty flag, not multiple ones, for WebGL (for now)
  Paintable.PaintableStatefulDrawable.mixin( Path.PathWebGLDrawable );
  SelfDrawable.Poolable.mixin( Path.PathWebGLDrawable ); // pooling

  /*---------------------------------------------------------------------------*
   * Pixi Rendering
   *----------------------------------------------------------------------------*/

  Path.PathPixiDrawable = function PathPixiDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( PixiSelfDrawable, Path.PathPixiDrawable, {
    initialize: function( renderer, instance ) {
      this.initializePixiSelfDrawable( renderer, instance, false ); // never keep paths

      if ( !this.displayObject ) {
        this.displayObject = new PIXI.Graphics();
      }

      return this;
    },

    updatePixiSelf: function( node, graphics ) {
      graphics.clear();

      var shape = node.shape;
      var i = 0;
      var segment;
      if ( shape !== null ) {
        if ( node.getStrokeColor() ) {
          graphics.lineStyle( 5, node.getStrokeColor().toNumber() );
        }
        if ( node.getFillColor() ) {
          graphics.beginFill( node.getFillColor().toNumber() );
        }
        for ( i = 0; i < shape.subpaths.length; i++ ) {
          var subpath = shape.subpaths[ i ];
          for ( var k = 0; k < subpath.segments.length; k++ ) {
            segment = subpath.segments[ k ];
            if ( i === 0 && k === 0 ) {
              graphics.moveTo( segment.start.x, segment.start.y );
            }
            else {
              graphics.lineTo( segment.start.x, segment.start.y );
            }

            if ( k === subpath.segments.length - 1 ) {
              graphics.lineTo( segment.end.x, segment.end.y );
            }
          }

          if ( subpath.isClosed() ) {
            segment = subpath.segments[ 0 ];
            graphics.lineTo( segment.start.x, segment.start.y );
          }
        }

        graphics.endFill();
      }
      // TODO: geometry

      //graphics.moveTo( 0, 0 );
      //graphics.lineTo( 100, 100 );
      //graphics.endFill();
    },

    // stateless dirty methods:
    markDirtyShape: function() { this.markPaintDirty(); }
  } );
  Paintable.PaintableStatelessDrawable.mixin( Path.PathPixiDrawable );
  SelfDrawable.Poolable.mixin( Path.PathPixiDrawable );

  return Path;
} );


