// Copyright 2002-2013, University of Colorado

/**
 * A circular node that inherits Path, and allows for optimized drawing,
 * and improved parameter handling.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );

  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Features = require( 'SCENERY/util/Features' );
  var Fillable = require( 'SCENERY/nodes/Fillable' );
  var Strokable = require( 'SCENERY/nodes/Strokable' );
  
  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMCircleElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory
  var keepSVGCircleElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory
  
  scenery.Circle = function Circle( radius, options ) {
    if ( typeof radius === 'object' ) {
      // allow new Circle( { radius: ... } )
      // the mutators will call invalidateCircle() and properly set the shape
      options = radius;
      this._radius = options.radius;
    } else {
      this._radius = radius;

      // ensure we have a parameter object
      options = options || {};

    }
    // fallback for non-canvas or non-svg rendering, and for proper bounds computation

    Path.call( this, null, options );
  };
  var Circle = scenery.Circle;

  inherit( Path, Circle, {
    getStrokeRendererBitmask: function() {
      var bitmask = Path.prototype.getStrokeRendererBitmask.call( this );
      if ( this.hasStroke() && !this.getStroke().isGradient && !this.getStroke().isPattern && this.getLineWidth() <= this.getRadius() ) {
        bitmask |= scenery.bitmaskSupportsDOM;
      }
      return bitmask;
    },
    
    getPathRendererBitmask: function() {
      return scenery.bitmaskSupportsCanvas | scenery.bitmaskSupportsSVG | scenery.bitmaskBoundsValid | ( Features.borderRadius ? scenery.bitmaskSupportsDOM : 0 );
    },
    
    invalidateCircle: function() {
      assert && assert( this._radius >= 0, 'A circle needs a non-negative radius' );
      
      // sets our 'cache' to null, so we don't always have to recompute our shape
      this._shape = null;
      
      // should invalidate the path and ensure a redraw
      this.invalidateShape();
    },
    
    createCircleShape: function() {
      return Shape.circle( 0, 0, this._radius );
    },
    
    intersectsBoundsSelf: function( bounds ) {
      // TODO: handle intersection with somewhat-infinite bounds!
      var x = Math.abs( bounds.centerX );
      var y = Math.abs( bounds.centerY );
      var halfWidth = bounds.maxX - x;
      var halfHeight = bounds.maxY - y;
      
      // too far to have a possible intersection
      if ( x > halfWidth + this._radius || y > halfHeight + this._radius ) {
        return false;
      }
      
      // guaranteed intersection
      if ( x <= halfWidth || y <= halfHeight ) {
        return true;
      }
      
      // corner case
      x -= halfWidth;
      y -= halfHeight;
      return x * x + y * y <= this._radius * this._radius;
    },
    
    paintCanvas: function( wrapper ) {
      var context = wrapper.context;
      
      context.beginPath();
      context.arc( 0, 0, this._radius, 0, Math.PI * 2, false );
      context.closePath();
      
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
    },
    
    // create a circle instead of a path, hopefully it is faster in implementations
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( scenery.svgns, 'circle' );
    },

    // optimized for the circle element instead of path
    updateSVGFragment: function( circle ) {
      circle.setAttribute( 'r', this._radius );

      circle.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },
    
    /*---------------------------------------------------------------------------*
     * DOM support
     *----------------------------------------------------------------------------*/
    
    domUpdateTransformOnRepaint: true, // since we have to integrate the baseline offset into the CSS transform, signal to DOMLayer
    
    getDOMElement: function() {
      var fill = document.createElement( 'div' );
      var stroke = document.createElement( 'div' );
      fill.appendChild( stroke );
      fill.style.display = 'block';
      fill.style.position = 'absolute';
      fill.style.left = '0';
      fill.style.top = '0';
      stroke.style.display = 'block';
      stroke.style.position = 'absolute';
      stroke.style.left = '0';
      stroke.style.top = '0';
      return fill;
    },

    updateDOMElement: function( fill ) {
      fill.style.width = ( 2 * this._radius ) + 'px';
      fill.style.height = ( 2 * this._radius ) + 'px';
      fill.style[Features.borderRadius] = this._radius + 'px';
      fill.style.backgroundColor = this.getCSSFill();
      
      var stroke = fill.childNodes[0];
      if ( this.hasStroke() ) {
        stroke.style.width = ( 2 * this._radius - this.getLineWidth() ) + 'px';
        stroke.style.height = ( 2 * this._radius - this.getLineWidth() ) + 'px';
        stroke.style.left = ( -this.getLineWidth() / 2 ) + 'px';
        stroke.style.top = ( -this.getLineWidth() / 2 ) + 'px';
        stroke.style.borderStyle = 'solid';
        stroke.style.borderColor = this.getSimpleCSSFill();
        stroke.style.borderWidth = this.getLineWidth() + 'px';
        stroke.style[Features.borderRadius] = ( this._radius + this.getLineWidth() / 2 ) + 'px';
      } else {
        stroke.style.borderStyle = 'none';
      }
    },
    
    // override the transform since we need to customize it with a DOM offset
    updateCSSTransform: function( transform, element ) {
      // shift the text vertically, postmultiplied with the entire transform.
      var matrix = transform.getMatrix().timesMatrix( Matrix3.translation( -this._radius, -this._radius ) );
      scenery.Util.applyCSSTransform( matrix, element );
    },
    
    createDOMState: function( domSelfDrawable ) {
      return Circle.CircleDOMState.createFromPool( domSelfDrawable );
    },
    
    createSVGState: function( svgSelfDrawable ) {
      return Circle.CircleSVGState.createFromPool( svgSelfDrawable );
    },

    getBasicConstructor: function( propLines ) {
      return 'new scenery.Circle( ' + this._radius + ', {' + propLines + '} )';
    },

    getRadius: function() {
      return this._radius;
    },

    setRadius: function( radius ) {
      assert && assert( typeof radius === 'number', 'Circle.radius must be a number' );
      
      if ( this._radius !== radius ) {
        this._radius = radius;
        this.invalidateCircle();
        
        var stateLen = this._visualStates.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._visualStates.markDirtyRadius();
        }
      }
      return this;
    },

    computeShapeBounds: function() {
      var bounds = new Bounds2( -this._radius, -this._radius, this._radius, this._radius );
      if ( this._stroke ) {
        // since we are axis-aligned, any stroke will expand our bounds by a guaranteed set amount
        bounds = bounds.dilated( this.getLineWidth() / 2 );
      }
      return bounds;
    },

    // accelerated hit detection
    containsPointSelf: function( point ) {
      var magSq = point.x * point.x + point.y * point.y;
      var result = true;
      var iRadius;
      if ( this._strokePickable ) {
        iRadius = this.getLineWidth() / 2;
        var outerRadius = this._radius + iRadius;
        result = result && magSq <= outerRadius * outerRadius;
      }
      
      if ( this._fillPickable ) {
        if ( this._strokePickable ) {
          // we were either within the outer radius, or not
          return result;
        } else {
          // just testing in the fill range
          return magSq <= this._radius * this._radius;
        }
      } else if ( this._strokePickable ) {
        var innerRadius = this._radius - iRadius;
        return result && magSq >= innerRadius * innerRadius;
      } else {
        return false; // neither stroke nor fill is pickable
      }
    },

    get radius() { return this.getRadius(); },
    set radius( value ) { return this.setRadius( value ); },
    
    setShape: function( shape ) {
      if ( shape !== null ) {
        throw new Error( 'Cannot set the shape of a scenery.Circle to something non-null' );
      } else {
        // probably called from the Path constructor
        this.invalidateShape();
      }
    },
    
    getShape: function() {
      if ( !this._shape ) {
        this._shape = this.createCircleShape();
      }
      return this._shape;
    },
    
    hasShape: function() {
      return true;
    }
  } );

  // not adding mutators for now
  Circle.prototype._mutatorKeys = [ 'radius' ].concat( Path.prototype._mutatorKeys );
  
  /*---------------------------------------------------------------------------*
  * Rendering State
  *----------------------------------------------------------------------------*/
  
  var CircleRenderState = Circle.CircleRenderState = function( drawable ) {
    // important to keep this in the constructor (so our hidden class works out nicely)
    this.initialize( drawable );
  };
  CircleRenderState.prototype = {
    constructor: CircleRenderState,
    
    // initializes, and resets (so we can support pooled states)
    initialize: function( drawable ) {
      // TODO: it's a bit weird to set it this way?
      drawable.visualState = this;
      
      this.drawable = drawable;
      this.node = drawable.node;
      
      this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
      this.dirtyRadius = true;   
      
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
      this.dirtyRadius = true;
      this.markPaintDirty();
    },
    setToClean: function() {
      this.paintDirty = false;
      this.dirtyRadius = false;
      
      this.cleanFillableState();
      this.cleanStrokableState();
    }
  };
  /* jshint -W064 */
  Fillable.FillableState( CircleRenderState );
  /* jshint -W064 */
  Strokable.StrokableState( CircleRenderState );
  
  /*---------------------------------------------------------------------------*
  * DOM rendering
  *----------------------------------------------------------------------------*/
  
  var CircleDOMState = Circle.CircleDOMState = function( drawable ) {
    // important to keep this in the constructor (so our hidden class works out nicely)
    this.initialize( drawable );
  };
  CircleDOMState.prototype = {
    constructor: CircleDOMState,
    
    // initializes, and resets (so we can support pooled states)
    initialize: function( drawable ) {
      CircleRenderState.prototype.initialize.call( this, drawable );
      
      this.transformDirty = true;
      this.forceAcceleration = false; // later changed by drawable if necessary
      
      if ( !this.matrix ) {
        this.matrix = Matrix3.dirtyFromPool();
      }
      
      // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
      // allocation and performance costs)
      if ( !this.fillElement || !this.strokeElement ) {
        var fillElement = this.fillElement = document.createElement( 'div' );
        var strokeElement = this.strokeElement = document.createElement( 'div' );
        fillElement.style.display = 'block';
        fillElement.style.position = 'absolute';
        fillElement.style.left = '0';
        fillElement.style.top = '0';
        strokeElement.style.display = 'block';
        strokeElement.style.position = 'absolute';
        strokeElement.style.left = '0';
        strokeElement.style.top = '0';
        fillElement.appendChild( strokeElement );
      }
      
      this.domElement = this.fillElement;
      
      return this; // allow for chaining
    },
    
    updateDOM: function() {
      var node = this.node;
      var fillElement = this.fillElement;
      var strokeElement = this.strokeElement;
      
      if ( this.paintDirty ) {
        if ( this.dirtyRadius ) {
          fillElement.style.width = ( 2 * this._radius ) + 'px';
          fillElement.style.height = ( 2 * this._radius ) + 'px';
          fillElement.style[Features.borderRadius] = this._radius + 'px';
        }
        if ( this.dirtyFill ) {
          fillElement.style.backgroundColor = this.getCSSFill();
        }
        
        if ( this.dirtyStroke ) {
          // update stroke presence
          if ( node.hasStroke() ) {
            strokeElement.style.borderStyle = 'solid';
          } else {
            strokeElement.style.borderStyle = 'none';
          }
        }
        
        if ( this.hasStroke() ) {
          // since we only execute these if we have a stroke, we need to redo everything if there was no stroke previously.
          // the other option would be to update stroked information when there is no stroke (major performance loss for fill-only Circles)
          var hadNoStrokeBefore = this.lastStroke === null;
          
          if ( hadNoStrokeBefore || this.dirtyLineWidth || this.dirtyRadius ) {
            strokeElement.style.width = ( 2 * this._radius - this.getLineWidth() ) + 'px';
            strokeElement.style.height = ( 2 * this._radius - this.getLineWidth() ) + 'px';
            strokeElement.style[Features.borderRadius] = ( this._radius + this.getLineWidth() / 2 ) + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyLineWidth ) {
            strokeElement.style.left = ( -this.getLineWidth() / 2 ) + 'px';
            strokeElement.style.top = ( -this.getLineWidth() / 2 ) + 'px';
            strokeElement.style.borderWidth = this.getLineWidth() + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyStroke ) {
            strokeElement.style.borderColor = this.getSimpleCSSFill();
          }
        }
      }
      
      // shift the element vertically, postmultiplied with the entire transform.
      if ( this.transformDirty || this.dirtyRadius ) {
        this.matrix.set( this.drawable.getTransformMatrix() );
        var translation = Matrix3.translation( -node._radius, -node._radius );
        this.matrix.multiplyMatrix( translation );
        translation.freeToPool();
        scenery.Util.applyCSSTransform( this.matrix, this.fillElement, this.forceAcceleration );
      }
      
      // clear all of the dirty flags
      this.setToClean();
    },
    
    // release the DOM elements from the poolable visual state so they aren't kept in memory. May not be done on platforms where we have enough memory to pool these
    onDetach: function() {
      if ( !keepDOMCircleElements ) {
        // clear the references
        this.fillElement = null;
        this.strokeElement = null;
        this.domElement = null;
      }
      
      // put us back in the pool
      this.freeToPool();
    },
    
    setToClean: function() {
      CircleRenderState.prototype.setToClean.call( this );
      
      this.transformDirty = false;
    }
  };
  /* jshint -W064 */
  Fillable.FillableState( CircleDOMState );
  /* jshint -W064 */
  Strokable.StrokableState( CircleDOMState );
  // for pooling, allow CircleDOMState.createFromPool( drawable ) and state.freeToPool(). Creation will initialize the state to the intial state
  /* jshint -W064 */
  Poolable( CircleDOMState, {
    defaultFactory: function() { return new CircleDOMState(); },
    constructorDuplicateFactory: function( pool ) {
      return function( drawable ) {
        if ( pool.length ) {
          return pool.pop().initialize( drawable );
        } else {
          return new CircleDOMState( drawable );
        }
      };
    }
  } );
  
  /*---------------------------------------------------------------------------*
  * SVG Rendering
  *----------------------------------------------------------------------------*/
  
  var CircleSVGState = Circle.CircleSVGState = inherit( CircleRenderState, function CircleSVGState( drawable ) {
    CircleRenderState.call( this, drawable );
  }, {
    initialize: function( drawable ) {
      CircleRenderState.prototype.initialize.call( this, drawable );
      
      this.defs = drawable.defs;
      
      // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
      // allocation and performance costs)
      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'circle' );
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
      var circle = this.svgElement;
      
      if ( this.paintDirty ) {
        if ( this.dirtyRadius ) {
          circle.setAttribute( 'r', this._radius );
        }
        
        // circle.setAttribute( 'style', node.getSVGFillStyle() + node.getSVGStrokeStyle() );
        
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
          circle.setAttribute( 'style', this.fillState.style + this.strokeState.baseStyle + this.strokeState.extraStyle );
        }
      }
      
      // clear all of the dirty flags
      this.setToClean();
    },
    
    // release the DOM elements from the poolable visual state so they aren't kept in memory. May not be done on platforms where we have enough memory to pool these
    onDetach: function() {
      if ( !keepSVGCircleElements ) {
        // clear the references
        this.svgElement = null;
      }
      
      this.fillState.dispose();
      
      // put us back in the pool
      this.freeToPool();
    },
    
    setToClean: function() {
      CircleRenderState.prototype.setToClean.call( this );
    }
  } );
  
  // for pooling, allow CircleSVGState.createFromPool( drawable ) and state.freeToPool(). Creation will initialize the state to the intial state
  /* jshint -W064 */
  Poolable( CircleSVGState, {
    defaultFactory: function() { return new CircleSVGState(); },
    constructorDuplicateFactory: function( pool ) {
      return function( drawable ) {
        if ( pool.length ) {
          return pool.pop().initialize( drawable );
        } else {
          return new CircleSVGState( drawable );
        }
      };
    }
  } );

  return Circle;
} );
