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
  var scenery = require( 'SCENERY/scenery' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Matrix3 = require( 'DOT/Matrix3' );

  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Features = require( 'SCENERY/util/Features' );
  var Fillable = require( 'SCENERY/nodes/Fillable' );
  var Strokable = require( 'SCENERY/nodes/Strokable' );
  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  
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
    
    //OHTWO @deprecated
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
    //OHTWO @deprecated
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( scenery.svgns, 'circle' );
    },

    // optimized for the circle element instead of path
    //OHTWO @deprecated
    updateSVGFragment: function( circle ) {
      circle.setAttribute( 'r', this._radius );

      circle.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },
    
    /*---------------------------------------------------------------------------*
     * DOM support
     *----------------------------------------------------------------------------*/
    
    //OHTWO @deprecated
    domUpdateTransformOnRepaint: true, // since we have to integrate the baseline offset into the CSS transform, signal to DOMLayer
    
    //OHTWO @deprecated
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
    
    //OHTWO @deprecated
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
        stroke.style.borderColor = this.getSimpleCSSStroke();
        stroke.style.borderWidth = this.getLineWidth() + 'px';
        stroke.style[Features.borderRadius] = ( this._radius + this.getLineWidth() / 2 ) + 'px';
      } else {
        stroke.style.borderStyle = 'none';
      }
    },
    
    // override the transform since we need to customize it with a DOM offset
    //OHTWO @deprecated
    updateCSSTransform: function( transform, element ) {
      // shift the text vertically, postmultiplied with the entire transform.
      var matrix = transform.getMatrix().timesMatrix( Matrix3.translation( -this._radius, -this._radius ) );
      scenery.Util.applyCSSTransform( matrix, element );
    },
    
    createDOMDrawable: function( renderer, instance ) {
      return Circle.CircleDOMDrawable.createFromPool( renderer, instance );
    },
    
    createSVGDrawable: function( renderer, instance ) {
      return Circle.CircleSVGDrawable.createFromPool( renderer, instance );
    },
    
    createCanvasDrawable: function( renderer, instance ) {
      return Circle.CircleCanvasDrawable.createFromPool( renderer, instance );
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
        
        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[i].markDirtyRadius();
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
  * Rendering State mixin (DOM/SVG)
  *----------------------------------------------------------------------------*/
  
  var CircleRenderState = Circle.CircleRenderState = function( drawableType ) {
    var proto = drawableType.prototype;
    
    // initializes, and resets (so we can support pooled states)
    proto.initializeState = function() {
      this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
      this.dirtyRadius = true;
      
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
    
    proto.markDirtyRadius = function() {
      this.dirtyRadius = true;
      this.markPaintDirty();
    };
    
    proto.setToCleanState = function() {
      this.paintDirty = false;
      this.dirtyRadius = false;
      
      this.cleanFillableState();
      this.cleanStrokableState();
    };
    
    /* jshint -W064 */
    Fillable.FillableState( drawableType );
    /* jshint -W064 */
    Strokable.StrokableState( drawableType );
  };
  
  /*---------------------------------------------------------------------------*
  * DOM rendering
  *----------------------------------------------------------------------------*/
  
  var CircleDOMDrawable = Circle.CircleDOMDrawable = inherit( DOMSelfDrawable, function CircleDOMDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  }, {
    // initializes, and resets (so we can support pooled states)
    initialize: function( renderer, instance ) {
      this.initializeDOMSelfDrawable( renderer, instance );
      this.initializeState();
      
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
          fillElement.style.width = ( 2 * node._radius ) + 'px';
          fillElement.style.height = ( 2 * node._radius ) + 'px';
          fillElement.style[Features.borderRadius] = node._radius + 'px';
        }
        if ( this.dirtyFill ) {
          fillElement.style.backgroundColor = node.getCSSFill();
        }
        
        if ( this.dirtyStroke ) {
          // update stroke presence
          if ( node.hasStroke() ) {
            strokeElement.style.borderStyle = 'solid';
          } else {
            strokeElement.style.borderStyle = 'none';
          }
        }
        
        if ( node.hasStroke() ) {
          // since we only execute these if we have a stroke, we need to redo everything if there was no stroke previously.
          // the other option would be to update stroked information when there is no stroke (major performance loss for fill-only Circles)
          var hadNoStrokeBefore = this.lastStroke === null;
          
          if ( hadNoStrokeBefore || this.dirtyLineWidth || this.dirtyRadius ) {
            strokeElement.style.width = ( 2 * node._radius - node.getLineWidth() ) + 'px';
            strokeElement.style.height = ( 2 * node._radius - node.getLineWidth() ) + 'px';
            strokeElement.style[Features.borderRadius] = ( node._radius + node.getLineWidth() / 2 ) + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyLineWidth ) {
            strokeElement.style.left = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.top = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.borderWidth = node.getLineWidth() + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyStroke ) {
            strokeElement.style.borderColor = node.getSimpleCSSStroke();
          }
        }
      }
      
      // shift the element vertically, postmultiplied with the entire transform.
      if ( this.transformDirty || this.dirtyRadius ) {
        this.matrix.set( this.getTransformMatrix() );
        var translation = Matrix3.translation( -node._radius, -node._radius );
        this.matrix.multiplyMatrix( translation );
        translation.freeToPool();
        scenery.Util.applyCSSTransform( this.matrix, this.fillElement, this.forceAcceleration );
      }
      
      // clear all of the dirty flags
      this.setToClean();
    },
    
    onAttach: function( node ) {
      
    },
    
    // release the DOM elements from the poolable visual state so they aren't kept in memory. May not be done on platforms where we have enough memory to pool these
    onDetach: function( node ) {
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
      this.setToCleanState();
      
      this.transformDirty = false;
    }
  } );

  /* jshint -W064 */
  CircleRenderState( CircleDOMDrawable );
  
  /* jshint -W064 */
  SelfDrawable.Poolable( CircleDOMDrawable );
  
  /*---------------------------------------------------------------------------*
  * SVG Rendering
  *----------------------------------------------------------------------------*/
  
  Circle.CircleSVGDrawable = SVGSelfDrawable.createDrawable( {
    type: function CircleSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    stateType: CircleRenderState,
    initialize: function( renderer, instance ) {
      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'circle' );
      }
    },
    updateSVG: function( node, circle ) {
      if ( this.dirtyRadius ) {
        circle.setAttribute( 'r', node._radius );
      }
      
      this.updateFillStrokeStyle( circle );
    },
    usesFill: true,
    usesStroke: true,
    keepElements: keepSVGCircleElements
  } );
  
  /*---------------------------------------------------------------------------*
  * Canvas rendering
  *----------------------------------------------------------------------------*/
  
  Circle.CircleCanvasDrawable = CanvasSelfDrawable.createDrawable( {
    type: function CircleCanvasDrawable( renderer, instance ) { this.initialize( renderer, instance ); },
    paintCanvas: function paintCanvasCircle( wrapper ) {
      var context = wrapper.context;
      var node = this.node;
      
      context.beginPath();
      context.arc( 0, 0, node._radius, 0, Math.PI * 2, false );
      context.closePath();
      
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
    },
    usesFill: true,
    usesStroke: true,
    dirtyMethods: ['markDirtyRadius']
  } );

  return Circle;
} );
