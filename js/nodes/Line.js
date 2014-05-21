// Copyright 2002-2014, University of Colorado

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
  
  var Fillable = require( 'SCENERY/nodes/Fillable' );
  var Strokable = require( 'SCENERY/nodes/Strokable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  
  // TODO: change this based on memory and performance characteristics of the platform
  var keepSVGLineElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory
  
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
      } else {
        // assumes Line( { ... } ), init to zero for now
        this._x1 = 0;
        this._y1 = 0;
        this._x2 = 0;
        this._y2 = 0;
        options = x1 || {};
      }
    } else {
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
    setLine: function( x1, y1, x2, y2 ) {
      assert && assert( x1 !== undefined && y1 !== undefined && x2 !== undefined && y2 !== undefined, 'parameters need to be defined' );
      
      this._x1 = x1;
      this._y1 = y1;
      this._x2 = x2;
      this._y2 = y2;
      this.invalidateLine();
    },
    
    setPoint1: function( x1, y1 ) {
      if ( typeof x1 === 'number' ) {
        // setPoint1( x1, y1 );
        this.setLine( x1, y1, this._x2, this._y2 );
      } else {
        // setPoint1( Vector2 )
        this.setLine( x1.x, x1.y, this._x2, this._y2 );
      }
    },
    set p1( point ) { this.setPoint1( point ); },
    get p1() { return new Vector2( this._x1, this._y1 ); },
    
    setPoint2: function( x2, y2 ) {
      if ( typeof x2 === 'number' ) {
        // setPoint2( x2, y2 );
        this.setLine( this._x1, this._y1, x2, y2 );
      } else {
        // setPoint2( Vector2 )
        this.setLine( this._x1, this._y1, x2.x, x2.y );
      }
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
      } else {
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
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Line( ' + this._x1 + ', ' + this._y1 + ', ' + this._x1 + ', ' + this._y1 + ', {' + propLines + '} )';
    },
    
    setShape: function( shape ) {
      if ( shape !== null ) {
        throw new Error( 'Cannot set the shape of a scenery.Line to something non-null' );
      } else {
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
    }
    
  } );
  
  function addLineProp( capitalizedShort ) {
    var lowerShort = capitalizedShort.toLowerCase();
    
    var getName = 'get' + capitalizedShort;
    var setName = 'set' + capitalizedShort;
    var privateName = '_' + lowerShort;
    
    Line.prototype[getName] = function() {
      return this[privateName];
    };
    
    Line.prototype[setName] = function( value ) {
      if ( this[privateName] !== value ) {
        this[privateName] = value;
        this.invalidateLine();
      }
      return this;
    };
    
    Object.defineProperty( Line.prototype, lowerShort, {
      set: Line.prototype[setName],
      get: Line.prototype[getName]
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
  
  var LineRenderState = Line.LineRenderState = function( drawableType ) {
    var proto = drawableType.prototype;
    
    // initializes, and resets (so we can support pooled states)
    proto.initializeState = function() {
      this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
      this.dirtyX1 = true;
      this.dirtyY1 = true;
      this.dirtyX2 = true;
      this.dirtyY2 = true;
      
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
  
  Line.LineSVGDrawable = SVGSelfDrawable.createDrawable( {
    type: function LineSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    stateType: LineRenderState,
    initialize: function( renderer, instance ) {
      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'line' );
      }
    },
    updateSVG: function( node, line ) {
      if ( this.dirtyX1 ) {
        line.setAttribute( 'x1', node._x1 );
      }
      if ( this.dirtyY1 ) {
        line.setAttribute( 'y1', node._y1 );
      }
      if ( this.dirtyX2 ) {
        line.setAttribute( 'x2', node._x2 );
      }
      if ( this.dirtyY2 ) {
        line.setAttribute( 'y2', node._y2 );
      }
      
      this.updateFillStrokeStyle( line );
    },
    usesFill: true, // NOTE: doesn't use fill, but for now developer option was that "we shouldn't error out when setting a fill on a Line"
    usesStroke: true,
    keepElements: keepSVGLineElements
  } );
  
  /*---------------------------------------------------------------------------*
  * Canvas rendering
  *----------------------------------------------------------------------------*/
  
  Line.LineCanvasDrawable = CanvasSelfDrawable.createDrawable( {
    type: function LineCanvasDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    paintCanvas: function paintCanvasLine( wrapper, node ) {
      var context = wrapper.context;
      
      context.beginPath();
      context.moveTo( node._x1, node._y1 );
      context.lineTo( node._x2, node._y2 );
      context.closePath();
      
      if ( node._stroke ) {
        node.beforeCanvasStroke( wrapper ); // defined in Strokable
        context.stroke();
        node.afterCanvasStroke( wrapper ); // defined in Strokable
      }
    },
    usesFill: true, // NOTE: doesn't use fill, but for now developer option was that "we shouldn't error out when setting a fill on a Line"
    usesStroke: true,
    dirtyMethods: ['markDirtyX1', 'markDirtyY1', 'markDirtyX2', 'markDirtyY2']
  } );
  
  return Line;
} );


