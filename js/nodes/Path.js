// Copyright 2002-2013, University of Colorado

/**
 * A Path draws a Shape with a specific type of fill and stroke.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var Shape = require( 'KITE/Shape' );
  
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );
  require( 'SCENERY/layers/Renderer' );
  var Fillable = require( 'SCENERY/nodes/Fillable' );
  var Strokable = require( 'SCENERY/nodes/Strokable' );
  
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
      this.markOldSelfPaint();
      
      this._strokedShape = null;
      
      if ( this.hasShape() ) {
        this.invalidateSelf( this.computeShapeBounds() );
        this.invalidatePaint();
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
    
    paintCanvas: function( wrapper ) {
      var context = wrapper.context;
      
      if ( this.hasShape() ) {
        // TODO: fill/stroke delay optimizations?
        context.beginPath();
        this._shape.writeToContext( context );

        if ( this._fill ) {
          this.beforeCanvasFill( wrapper ); // defined in Fillable
          context.fill();
          this.afterCanvasFill( wrapper ); // defined in Fillable
        }
        if ( this._stroke ) {
          this.beforeCanvasStroke( wrapper ); // defined in Strokable
          context.stroke();
          this.afterCanvasStroke( wrapper ); // defined in Strokable
        }
      }
    },
    
    paintWebGL: function( state ) {
      throw new Error( 'Path.prototype.paintWebGL unimplemented' );
    },
    
    // svg element, the <defs> block, and the associated group for this node's transform
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( scenery.svgns, 'path' );
    },
    
    updateSVGFragment: function( path ) {
      var svgPath = this.hasShape() ? this._shape.getSVGPath() : "";
      
      // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
      // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
      // the attribute can cause this bug
      if ( !svgPath ) { svgPath = 'M0 0'; }
      
      if ( svgPath ) {
        // only set the SVG path if it's not the empty string
        path.setAttribute( 'd', svgPath );
      } else if ( path.hasAttribute( 'd' ) ) {
        path.removeAttribute( 'd' );
      }
      
      path.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },
    
    // support patterns, gradients, and anything else we need to put in the <defs> block
    updateSVGDefs: function( svg, defs ) {
      // remove old definitions if they exist
      this.removeSVGDefs( svg, defs );
      
      // add new ones if applicable
      this.addSVGFillDef( svg, defs );
      this.addSVGStrokeDef( svg, defs );
      
      assert && assert( !this.requiresSVGBoundsWorkaround(), 'No workaround for https://github.com/phetsims/scenery/issues/196 is provided at this time, please add an epsilon' );
    },
    
    // cleans up references created with udpateSVGDefs()
    removeSVGDefs: function( svg, defs ) {
      this.removeSVGFillDef( svg, defs );
      this.removeSVGStrokeDef( svg, defs );
    },
    
    createSVGState: function( svgSelfDrawable ) {
      return Path.PathSVGState.createFromPool( svgSelfDrawable );
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
  * Rendering State
  *----------------------------------------------------------------------------*/
  
  var PathRenderState = Path.PathRenderState = function( drawable ) {
    // important to keep this in the constructor (so our hidden class works out nicely)
    this.initialize( drawable );
  };
  PathRenderState.prototype = {
    constructor: PathRenderState,
    
    // initializes, and resets (so we can support pooled states)
    initialize: function( drawable ) {
      // TODO: it's a bit weird to set it this way?
      drawable.visualState = this;
      
      this.drawable = drawable;
      this.node = drawable.node;
      
      this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
      this.dirtyShape = true;   
      
      // adds fill/stroke-specific flags and state
      this.initializeFillableState();
      this.initializeStrokableState();
      
      return this; // allow for chaining
    },
    
    // catch-all dirty, if anything that isn't a transform is marked as dirty
    markPaintDirty: function() {
      this.paintDirty = true;
      this.drawable.markDirty();
    },
    markDirtyRadius: function() {
      this.dirtyShape = true;
      this.markPaintDirty();
    },
    setToClean: function() {
      this.paintDirty = false;
      this.dirtyShape = false;
      
      this.cleanFillableState();
      this.cleanStrokableState();
    }
  };
  /* jshint -W064 */
  Fillable.FillableState( PathRenderState );
  /* jshint -W064 */
  Strokable.StrokableState( PathRenderState );
  
  /*---------------------------------------------------------------------------*
  * SVG Rendering
  *----------------------------------------------------------------------------*/
  
  var PathSVGState = Path.PathSVGState = inherit( PathRenderState, function PathSVGState( drawable ) {
    PathRenderState.call( this, drawable );
  }, {
    initialize: function( drawable ) {
      PathRenderState.prototype.initialize.call( this, drawable );
      
      this.defs = drawable.defs;
      
      // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
      // allocation and performance costs)
      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'path' );
      }
      
      if ( !this.fillState ) {
        this.fillState = new Fillable.FillSVGState();
      } else {
        this.fillState.initialize();
      }
      
      if ( !this.strokeState ) {
        this.strokeState = new Strokable.StrokeSVGState();
      } else {
        this.strokeState.initialize();
      }
      
      return this; // allow for chaining
    },
    
    updateDefs: function( defs ) {
      this.defs = defs;
      this.fillState.updateDefs( defs );
      this.strokeState.updateDefs( defs );
    },
    
    updateSVG: function() {
      var node = this.node;
      var path = this.svgElement;
      
      assert && assert( !this.requiresSVGBoundsWorkaround(), 'No workaround for https://github.com/phetsims/scenery/issues/196 is provided at this time, please add an epsilon' );
      
      if ( this.paintDirty ) {
        if ( this.dirtyShape ) {
          var svgPath = this.hasShape() ? this._shape.getSVGPath() : '';
          
          // temporary workaround for https://bugs.webkit.org/show_bug.cgi?id=78980
          // and http://code.google.com/p/chromium/issues/detail?id=231626 where even removing
          // the attribute can cause this bug
          if ( !svgPath ) { svgPath = 'M0 0'; }
          
          // only set the SVG path if it's not the empty string
          path.setAttribute( 'd', svgPath );
        }
        
        // path.setAttribute( 'style', node.getSVGFillStyle() + node.getSVGStrokeStyle() );
        
        if ( this.dirtyFill ) {
          this.fillState.updateFill( this.defs, node._fill );
        }
        if ( this.dirtyStroke ) {
          this.strokeState.updateStroke( this.defs, node._stroke );
        }
        var strokeParameterDirty = this.dirtyLineWidth || this.dirtyLineOptions;
        if ( strokeParameterDirty ) {
          this.strokeState.updateStrokeParameters( node );
        }
        if ( this.dirtyFill || this.dirtyStroke || strokeParameterDirty ) {
          path.setAttribute( 'style', this.fillState.style + this.strokeState.baseStyle + this.strokeState.extraStyle );
        }
      }
      
      // clear all of the dirty flags
      this.setToClean();
    },
    
    // release the DOM elements from the poolable visual state so they aren't kept in memory. May not be done on platforms where we have enough memory to pool these
    onDetach: function() {
      if ( !keepSVGPathElements ) {
        // clear the references
        this.svgElement = null;
      }
      
      this.fillState.dispose();
      
      // put us back in the pool
      this.freeToPool();
    },
    
    setToClean: function() {
      PathRenderState.prototype.setToClean.call( this );
    }
  } );
  
  // for pooling, allow PathSVGState.createFromPool( drawable ) and state.freeToPool(). Creation will initialize the state to the intial state
  /* jshint -W064 */
  Poolable( PathSVGState, {
    defaultFactory: function() { return new PathSVGState(); },
    constructorDuplicateFactory: function( pool ) {
      return function( drawable ) {
        if ( pool.length ) {
          return pool.pop().initialize( drawable );
        } else {
          return new PathSVGState( drawable );
        }
      };
    }
  } );
  
  return Path;
} );


