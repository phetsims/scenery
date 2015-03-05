// Copyright 2002-2014, University of Colorado Boulder

/**
 * A line that inherits Path, and allows for optimized drawing,
 * and improved line handling.
 *
 * TODO: add DOM support
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var KiteLine = require( 'KITE/segments/Line' );

  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Vector2 = require( 'DOT/Vector2' );

  var Paintable = require( 'SCENERY/nodes/Paintable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );

  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var PixiSelfDrawable = require( 'SCENERY/display/PixiSelfDrawable' );
  var SquareUnstrokedRectangle = require( 'SCENERY/display/webgl/SquareUnstrokedRectangle' );
  var Color = require( 'SCENERY/util/Color' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepSVGLineElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory
  var keepPixiLineElements = true; // whether we should pool Pixi elements for the SVG rendering states, or whether we should free them when possible for memory

  /**
   * Currently, all numerical parameters should be finite.
   * x1:         x-position of the start
   * y1:         y-position of the start
   * x2:         x-position of the end
   * y2:         y-position of the end
   *
   * Available constructors:
   * new Line( x1, y1, x2, y2, { ... } )
   * new Line( new Vector2( x1, y1 ), new Vector2( x2, y2 ), { ... } )
   * new Line( { x1: x1, y1: y1, x2: x2, y2: y2,  ... } )
   */
  scenery.Line = function Line( x1, y1, x2, y2, options ) {
    if ( typeof x1 === 'object' ) {
      if ( x1 instanceof Vector2 ) {
        // assumes Line( Vector2, Vector2, options );
        this._x1 = x1.x;
        this._y1 = x1.y;
        this._x2 = y1.x;
        this._y2 = y1.y;
        options = x2 || {};
      }
      else {
        // assumes Line( { ... } ), init to zero for now
        this._x1 = 0;
        this._y1 = 0;
        this._x2 = 0;
        this._y2 = 0;
        options = x1 || {};
      }
    }
    else {
      // new Line(  x1, y1, x2, y2, [options] )
      this._x1 = x1;
      this._y1 = y1;
      this._x2 = x2;
      this._y2 = y2;

      // ensure we have a parameter object
      options = options || {};
    }
    // fallback for non-canvas or non-svg rendering, and for proper bounds computation

    Path.call( this, null, options );
  };
  var Line = scenery.Line;

  inherit( Path, Line, {

    /**
     * Set the geometry of the line, including stand and end point.
     * @param {number} x1 - the start x coordinate
     * @param {number} y1 - the start y coordinate
     * @param {number} x2 - the end x coordinate
     * @param {number} y2 - the end y coordinate
     */
    setLine: function( x1, y1, x2, y2 ) {
      assert && assert( x1 !== undefined && y1 !== undefined && x2 !== undefined && y2 !== undefined, 'parameters need to be defined' );

      this._x1 = x1;
      this._y1 = y1;
      this._x2 = x2;
      this._y2 = y2;

      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        var state = this._drawables[ i ];
        state.markDirtyLine();
      }

      this.invalidateLine();
    },

    setPoint1: function( x1, y1 ) {
      if ( typeof x1 === 'number' ) {
        // setPoint1( x1, y1 );
        assert && assert( x1 !== undefined && y1 !== undefined, 'parameters need to be defined' );
        this._x1 = x1;
        this._y1 = y1;
      }
      else {
        // setPoint1( Vector2 )
        assert && assert( x1.x !== undefined && x1.y !== undefined, 'parameters need to be defined' );
        this._x1 = x1.x;
        this._y1 = x1.y;
      }
      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        var state = this._drawables[ i ];
        state.markDirtyP1();
      }
      this.invalidateLine();
    },
    set p1( point ) { this.setPoint1( point ); },
    get p1() { return new Vector2( this._x1, this._y1 ); },

    setPoint2: function( x2, y2 ) {
      if ( typeof x2 === 'number' ) {
        // setPoint2( x2, y2 );
        assert && assert( x2 !== undefined && y2 !== undefined, 'parameters need to be defined' );
        this._x2 = x2;
        this._y2 = y2;
      }
      else {
        // setPoint2( Vector2 )
        assert && assert( x2.x !== undefined && x2.y !== undefined, 'parameters need to be defined' );
        this._x2 = x2.x;
        this._y2 = x2.y;
      }
      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        var state = this._drawables[ i ];
        state.markDirtyP2();
      }
      this.invalidateLine();
    },
    set p2( point ) { this.setPoint2( point ); },
    get p2() { return new Vector2( this._x2, this._y2 ); },

    createLineShape: function() {
      return Shape.lineSegment( this._x1, this._y1, this._x2, this._y2 );
    },

    invalidateLine: function() {
      assert && assert( isFinite( this._x1 ), 'A rectangle needs to have a finite x1 (' + this._x1 + ')' );
      assert && assert( isFinite( this._y1 ), 'A rectangle needs to have a finite y1 (' + this._y1 + ')' );
      assert && assert( isFinite( this._x2 ), 'A rectangle needs to have a finite x2 (' + this._x2 + ')' );
      assert && assert( isFinite( this._y2 ), 'A rectangle needs to have a finite y2 (' + this._y2 + ')' );

      // sets our 'cache' to null, so we don't always have to recompute our shape
      this._shape = null;

      // should invalidate the path and ensure a redraw
      this.invalidateShape();
    },

    containsPointSelf: function( point ) {
      if ( this._strokePickable ) {
        return Path.prototype.containsPointSelf.call( this, point );
      }
      else {
        return false; // nothing is in a line! (although maybe we should handle edge points properly?)
      }
    },

    intersectsBoundsSelf: function( bounds ) {
      // TODO: optimization
      return new KiteLine( this.p1, this.p2 ).intersectsBounds( bounds );
    },

    canvasPaintSelf: function( wrapper ) {
      Line.LineCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    computeShapeBounds: function() {
      return Path.prototype.computeShapeBounds.call( this );
    },

    createSVGDrawable: function( renderer, instance ) {
      return Line.LineSVGDrawable.createFromPool( renderer, instance );
    },

    createCanvasDrawable: function( renderer, instance ) {
      return Line.LineCanvasDrawable.createFromPool( renderer, instance );
    },

    createWebGLDrawable: function( renderer, instance ) {
      return Line.LineWebGLDrawable.createFromPool( renderer, instance );
    },

    createPixiDrawable: function( renderer, instance ) {
      return Line.LinePixiDrawable.createFromPool( renderer, instance );
    },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Line( ' + this._x1 + ', ' + this._y1 + ', ' + this._x1 + ', ' + this._y1 + ', {' + propLines + '} )';
    },

    setShape: function( shape ) {
      if ( shape !== null ) {
        throw new Error( 'Cannot set the shape of a scenery.Line to something non-null' );
      }
      else {
        // probably called from the Path constructor
        this.invalidateShape();
      }
    },

    getShape: function() {
      if ( !this._shape ) {
        this._shape = this.createLineShape();
      }
      return this._shape;
    },

    hasShape: function() {
      return true;
    },

    // A line does not render its fill, so it supports all renderers.  Right?
    // - SR, 2014
    getFillRendererBitmask: function() {
      return scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG | scenery.bitmaskSupportsDOM | scenery.bitmaskSupportsWebGL | scenery.bitmaskSupportsPixi;
    }

  } );

  function addLineProp( capitalizedShort ) {
    var lowerShort = capitalizedShort.toLowerCase();

    var getName = 'get' + capitalizedShort;
    var setName = 'set' + capitalizedShort;
    var privateName = '_' + lowerShort;
    var dirtyMethodName = 'markDirty' + capitalizedShort;

    Line.prototype[ getName ] = function() {
      return this[ privateName ];
    };

    Line.prototype[ setName ] = function( value ) {
      if ( this[ privateName ] !== value ) {
        this[ privateName ] = value;
        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          var state = this._drawables[ i ];
          state[ dirtyMethodName ]();
        }
        this.invalidateLine();
      }
      return this;
    };

    Object.defineProperty( Line.prototype, lowerShort, {
      set: Line.prototype[ setName ],
      get: Line.prototype[ getName ]
    } );
  }

  addLineProp( 'X1' );
  addLineProp( 'Y1' );
  addLineProp( 'X2' );
  addLineProp( 'Y2' );

  // not adding mutators for now
  Line.prototype._mutatorKeys = [ 'p1', 'p2', 'x1', 'y1', 'x2', 'y2' ].concat( Path.prototype._mutatorKeys );

  /*---------------------------------------------------------------------------*
   * Rendering State mixin (DOM/SVG)
   *----------------------------------------------------------------------------*/

  Line.LineStatefulDrawable = {
    mixin: function( drawableType ) {
      var proto = drawableType.prototype;

      // initializes, and resets (so we can support pooled states)
      proto.initializeState = function() {
        this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
        this.dirtyX1 = true;
        this.dirtyY1 = true;
        this.dirtyX2 = true;
        this.dirtyY2 = true;

        // adds fill/stroke-specific flags and state
        this.initializePaintableState();

        return this; // allow for chaining
      };

      // catch-all dirty, if anything that isn't a transform is marked as dirty
      proto.markPaintDirty = function() {
        this.paintDirty = true;
        this.markDirty();
      };

      proto.markDirtyLine = function() {
        this.dirtyX1 = true;
        this.dirtyY1 = true;
        this.dirtyX2 = true;
        this.dirtyY2 = true;
        this.markPaintDirty();
      };

      proto.markDirtyP1 = function() {
        this.dirtyX1 = true;
        this.dirtyY1 = true;
        this.markPaintDirty();
      };

      proto.markDirtyP2 = function() {
        this.dirtyX2 = true;
        this.dirtyY2 = true;
        this.markPaintDirty();
      };

      proto.markDirtyX1 = function() {
        this.dirtyX1 = true;
        this.markPaintDirty();
      };

      proto.markDirtyY1 = function() {
        this.dirtyY1 = true;
        this.markPaintDirty();
      };

      proto.markDirtyX2 = function() {
        this.dirtyX2 = true;
        this.markPaintDirty();
      };

      proto.markDirtyY2 = function() {
        this.dirtyY2 = true;
        this.markPaintDirty();
      };

      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyX1 = false;
        this.dirtyY1 = false;
        this.dirtyX2 = false;
        this.dirtyY2 = false;

        this.cleanPaintableState();
      };

      Paintable.PaintableStatefulDrawable.mixin( drawableType );
    }
  };

  /*---------------------------------------------------------------------------*
   * Stateless drawable mixin
   *----------------------------------------------------------------------------*/

  Line.LineStatelessDrawable = {
    mixin: function( drawableType ) {
      var proto = drawableType.prototype;

      // initializes, and resets (so we can support pooled states)
      proto.initializeLineStateless = function() {
        this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
        return this; // allow for chaining
      };

      // catch-all dirty, if anything that isn't a transform is marked as dirty
      proto.markPaintDirty = function() {
        this.paintDirty = true;
        this.markDirty();
      };

      proto.markDirtyLine = function() {
        this.markPaintDirty();
      };

      proto.markDirtyP1 = function() {
        this.markPaintDirty();
      };

      proto.markDirtyP2 = function() {
        this.markPaintDirty();
      };

      proto.markDirtyX1 = function() {
        this.markPaintDirty();
      };

      proto.markDirtyY1 = function() {
        this.markPaintDirty();
      };

      proto.markDirtyX2 = function() {
        this.markPaintDirty();
      };

      proto.markDirtyY2 = function() {
        this.markPaintDirty();
      };

      Paintable.PaintableStatefulDrawable.mixin( drawableType );
    }
  };

  /*---------------------------------------------------------------------------*
   * SVG Rendering
   *----------------------------------------------------------------------------*/

  Line.LineSVGDrawable = function LineSVGDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( SVGSelfDrawable, Line.LineSVGDrawable, {
    initialize: function( renderer, instance ) {
      this.initializeSVGSelfDrawable( renderer, instance, true, keepSVGLineElements ); // usesPaint: true

      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'line' );
      }

      return this;
    },

    updateSVGSelf: function() {
      var line = this.svgElement;

      if ( this.dirtyX1 ) {
        line.setAttribute( 'x1', this.node._x1 );
      }
      if ( this.dirtyY1 ) {
        line.setAttribute( 'y1', this.node._y1 );
      }
      if ( this.dirtyX2 ) {
        line.setAttribute( 'x2', this.node._x2 );
      }
      if ( this.dirtyY2 ) {
        line.setAttribute( 'y2', this.node._y2 );
      }

      this.updateFillStrokeStyle( line );
    }
  } );
  Line.LineStatefulDrawable.mixin( Line.LineSVGDrawable );
  SelfDrawable.Poolable.mixin( Line.LineSVGDrawable );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  Line.LineCanvasDrawable = function LineCanvasDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( CanvasSelfDrawable, Line.LineCanvasDrawable, {
    initialize: function( renderer, instance ) {
      return this.initializeCanvasSelfDrawable( renderer, instance );
    },

    paintCanvas: function( wrapper, node ) {
      var context = wrapper.context;

      context.beginPath();
      context.moveTo( node._x1, node._y1 );
      context.lineTo( node._x2, node._y2 );
      context.closePath();

      if ( node._stroke ) {
        node.beforeCanvasStroke( wrapper ); // defined in Paintable
        context.stroke();
        node.afterCanvasStroke( wrapper ); // defined in Paintable
      }
    },

    // stateless dirty methods:
    markDirtyLine: function() { this.markPaintDirty(); },
    markDirtyP1: function() { this.markPaintDirty(); },
    markDirtyP2: function() { this.markPaintDirty(); },
    markDirtyX1: function() { this.markPaintDirty(); },
    markDirtyY1: function() { this.markPaintDirty(); },
    markDirtyX2: function() { this.markPaintDirty(); },
    markDirtyY2: function() { this.markPaintDirty(); }
  } );
  Paintable.PaintableStatelessDrawable.mixin( Line.LineCanvasDrawable );
  SelfDrawable.Poolable.mixin( Line.LineCanvasDrawable );

  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  Line.LineWebGLDrawable = inherit( WebGLSelfDrawable, function LineWebGLDrawable( renderer, instance ) {
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
      this.updateLine();

      //TODO: Update the state in the buffer arrays
    },

    //Nothing necessary since everything currently handled in the uModelViewMatrix below
    //However, we may switch to dynamic draw, and handle the matrix change only where necessary in the future?
    updateLine: function() {

      // TODO: a way to update the ColorTriangleBufferData.

      // TODO: move to PaintableWebGLState???
      if ( this.dirtyFill ) {
        this.color = Color.toColor( this.node._fill || 'red' );
        this.cleanPaintableState();
      }
      this.rectangleHandle.update();

      // TODO: Batch these updates?
      this.webglBlock.webGLRenderer.colorTriangleRenderer.updateTriangleBuffer( this.rectangleHandle );
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

    markDirtyLine: function() {
      this.markDirty();
    },

    markDirtyX1: function() {
      this.markDirty();
    },
    markDirtyY1: function() {
      this.markDirty();
    },
    markDirtyX2: function() {
      this.markDirty();
    },
    markDirtyY2: function() {
      this.markDirty();
    },

    // general flag set on the state, which we forward directly to the drawable's paint flag
    markPaintDirty: function() {
      this.markDirty();
    },

    //TODO: Make sure all of the dirty flags make sense here.  Should we be using fillDirty, paintDirty, dirty, etc?
    update: function() {
      if ( this.dirty ) {
        this.updateLine();
        this.dirty = false;
      }
    }
  } );
  // include stubs (stateless) for marking dirty stroke and fill (if necessary). we only want one dirty flag, not multiple ones, for WebGL (for now)
  Paintable.PaintableStatefulDrawable.mixin( Line.LineWebGLDrawable );
  SelfDrawable.Poolable.mixin( Line.LineWebGLDrawable ); // pooling

  /*---------------------------------------------------------------------------*
   * Pixi Rendering
   *----------------------------------------------------------------------------*/

  Line.LinePixiDrawable = function LinePixiDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( PixiSelfDrawable, Line.LinePixiDrawable, {
    initialize: function( renderer, instance ) {
      this.initializePixiSelfDrawable( renderer, instance, keepPixiLineElements );

      if ( !this.displayObject ) {
        this.displayObject = new PIXI.Graphics();
      }

      return this;
    },

    updatePixiSelf: function( node, graphics ) {
      this.displayObject.clear();
      if ( node.getStrokeColor() ) {
        graphics.lineStyle( node.lineWidth, node.getStrokeColor().toNumber() );
      }
      graphics.moveTo( node._x1, node._y1 );
      graphics.lineTo( node._x2, node._y2 );
    },

    // stateless dirty methods:
    markDirtyLine: function() { this.markPaintDirty(); },
    markDirtyP1: function() { this.markPaintDirty(); },
    markDirtyP2: function() { this.markPaintDirty(); },
    markDirtyX1: function() { this.markPaintDirty(); },
    markDirtyY1: function() { this.markPaintDirty(); },
    markDirtyX2: function() { this.markPaintDirty(); },
    markDirtyY2: function() { this.markPaintDirty(); }
  } );
Paintable.PaintableStatelessDrawable.mixin( Line.LinePixiDrawable );
  SelfDrawable.Poolable.mixin( Line.LinePixiDrawable );

  return Line;
} );


