// Copyright 2013-2015, University of Colorado Boulder

/**
 * A circular node that inherits Path, and allows for optimized drawing and improved parameter handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
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
  var Paintable = require( 'SCENERY/nodes/Paintable' );
  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  require( 'SCENERY/util/Util' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMCircleElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory
  var keepSVGCircleElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  /**
   * @constructor
   *
   * NOTE: There are two ways of invoking the constructor:
   * - new Circle( radius, { ... } )
   * - new Circle( { radius: radius, ... } )
   *
   * This allows the radius to be included in the parameter object for when that is convenient.
   *
   * @param {number} radius - The (non-negative) radius of the circle
   * @param {Object} [options] - Can contain Node's options, and/or CanvasNode options (e.g. canvasBound)
   */
  function Circle( radius, options ) {
    if ( typeof radius === 'object' ) {
      // allow new Circle( { radius: ... } )
      // the mutators will call invalidateCircle() and properly set the shape
      options = radius;
      this._radius = options.radius;
    }
    else {
      this._radius = radius;

      // ensure we have a parameter object
      options = options || {};

    }

    Path.call( this, null, options );
  }

  scenery.register( 'Circle', Circle );

  inherit( Path, Circle, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: [ 'radius' ].concat( Path.prototype._mutatorKeys ),

    /**
     * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
     *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
     *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
     * @public (scenery-internal)
     * @override
     */
    drawableMarkFlags: Path.prototype.drawableMarkFlags.concat( [ 'radius' ] ).filter( function( flag ) {
      // We don't want the shape flag, as that won't be called for Path subtypes.
      return flag !== 'shape';
    } ),

    /**
     * Determines the default allowed renderers (returned via the Renderer bitmask) that are allowed, given the
     * current stroke options.
     * @public (scenery-internal)
     * @override
     *
     * We can support the DOM renderer if there is a solid-styled stroke (which otherwise wouldn't be supported).
     *
     * @returns {number} - Renderer bitmask, see Renderer for details
     */
    getStrokeRendererBitmask: function() {
      var bitmask = Path.prototype.getStrokeRendererBitmask.call( this );
      if ( this.hasStroke() && !this.getStroke().isGradient && !this.getStroke().isPattern && this.getLineWidth() <= this.getRadius() ) {
        bitmask |= Renderer.bitmaskDOM;
      }
      return bitmask;
    },

    /**
     * Determines the allowed renderers that are allowed (or excluded) based on the current Path.
     * @public (scenery-internal)
     * @override
     *
     * @returns {number} - Renderer bitmask, see Renderer for details
     */
    getPathRendererBitmask: function() {
      // If we can use CSS borderRadius, we can support the DOM renderer.
      return Renderer.bitmaskCanvas | Renderer.bitmaskSVG | ( Features.borderRadius ? Renderer.bitmaskDOM : 0 );
    },

    /**
     * Notifies that the circle has changed (probably the radius), and invalidates path information and our cached
     * shape.
     * @private
     */
    invalidateCircle: function() {
      assert && assert( this._radius >= 0, 'A circle needs a non-negative radius' );

      // sets our 'cache' to null, so we don't always have to recompute our shape
      this._shape = null;

      // should invalidate the path and ensure a redraw
      this.invalidatePath();
    },

    /**
     * Returns a Shape that is equivalent to our rendered display. Generally used to lazily create a Shape instance
     * when one is needed, without having to do so beforehand.
     * @private
     *
     * @returns {Shape}
     */
    createCircleShape: function() {
      return Shape.circle( 0, 0, this._radius ).makeImmutable();
    },

    /**
     * Returns whether this Circle's selfBounds is intersected by the specified bounds.
     * @public
     *
     * @param {Bounds2} bounds - Bounds to test, assumed to be in the local coordinate frame.
     * @returns {boolean}
     */
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

    /**
     * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
     * coordinate frame of this node.
     * @protected
     * @override
     *
     * @param {CanvasContextWrapper} wrapper
     */
    canvasPaintSelf: function( wrapper ) {
      Circle.CircleCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    /**
     * Creates a DOM drawable for this Circle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {DOMSelfDrawable}
     */
    createDOMDrawable: function( renderer, instance ) {
      return Circle.CircleDOMDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a SVG drawable for this Circle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {SVGSelfDrawable}
     */
    createSVGDrawable: function( renderer, instance ) {
      return Circle.CircleSVGDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a Canvas drawable for this Circle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {CanvasSelfDrawable}
     */
    createCanvasDrawable: function( renderer, instance ) {
      return Circle.CircleCanvasDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a WebGL drawable for this Circle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {WebGLSelfDrawable}
     */
    createWebGLDrawable: function( renderer, instance ) {
      return Circle.CircleWebGLDrawable.createFromPool( renderer, instance );
    },

    /**
     * Returns a string containing constructor information for Node.string().
     * @protected
     * @override
     *
     * @param {string} propLines - A string representing the options properties that need to be set.
     * @returns {string}
     */
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Circle( ' + this._radius + ', {' + propLines + '} )';
    },

    /**
     * Sets the radius of the circle.
     * @public
     *
     * @param {number} radius
     * @returns {Circle} - 'this' reference, for chaining
     */
    setRadius: function( radius ) {
      assert && assert( typeof radius === 'number', 'Circle.radius must be a number' );
      assert && assert( radius >= 0, 'A circle needs a non-negative radius' );
      assert && assert( isFinite( radius ), 'A circle needs a finite radius' );

      if ( this._radius !== radius ) {
        this._radius = radius;
        this.invalidateCircle();

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyRadius();
        }
      }
      return this;
    },
    set radius( value ) { return this.setRadius( value ); },

    /**
     * Returns the radius of the circle.
     * @public
     *
     * @returns {number} - The radius of the circle
     */
    getRadius: function() {
      return this._radius;
    },
    get radius() { return this.getRadius(); },

    /**
     * Computes the bounds of the Circle, including any applied stroke. Overridden for efficiency.
     * @public
     * @override
     *
     * @returns {Bounds2}
     */
    computeShapeBounds: function() {
      var bounds = new Bounds2( -this._radius, -this._radius, this._radius, this._radius );
      if ( this._stroke ) {
        // since we are axis-aligned, any stroke will expand our bounds by a guaranteed set amount
        bounds = bounds.dilated( this.getLineWidth() / 2 );
      }
      return bounds;
    },

    /**
     * Computes whether the provided point is "inside" (contained) in this Circle's self content, or "outside".
     * @protected
     * @override
     *
     * Exists to optimize hit detection, as it's quick to compute for circles.
     *
     * @param {Vector2} point - Considered to be in the local coordinate frame
     * @returns {boolean}
     */
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
        }
        else {
          // just testing in the fill range
          return magSq <= this._radius * this._radius;
        }
      }
      else if ( this._strokePickable ) {
        var innerRadius = this._radius - iRadius;
        return result && magSq >= innerRadius * innerRadius;
      }
      else {
        return false; // neither stroke nor fill is pickable
      }
    },

    /**
     * It is impossible to set another shape on this Path subtype, as its effective shape is determined by other
     * parameters.
     * @public
     * @override
     *
     * @param {Shape|null} Shape - Throws an error if it is not null.
     */
    setShape: function( shape ) {
      if ( shape !== null ) {
        throw new Error( 'Cannot set the shape of a scenery.Circle to something non-null' );
      }
      else {
        // probably called from the Path constructor
        this.invalidatePath();
      }
    },

    /**
     * Returns an immutable copy of this Path subtype's representation.
     * @public
     * @override
     *
     * NOTE: This is created lazily, so don't call it if you don't have to!
     *
     * @returns {Shape}
     */
    getShape: function() {
      if ( !this._shape ) {
        this._shape = this.createCircleShape();
      }
      return this._shape;
    },

    /**
     * Returns whether this Path has an associated Shape (instead of no shape, represented by null)
     * @public
     * @override
     *
     * @returns {boolean}
     */
    hasShape: function() {
      // Always true for this Path subtype
      return true;
    }
  } );

  /*---------------------------------------------------------------------------*
   * Rendering State mixin (DOM/SVG)
   *----------------------------------------------------------------------------*/

  /**
   * A mixin to drawables for Circle that need to store state about what the current display is currently showing,
   * so that updates to the Circle will only be made on attributes that specifically changed (and no change will be
   * necessary for an attribute that changed back to its original/currently-displayed value). Generally, this is used
   * for DOM and SVG drawables.
   */
  Circle.CircleStatefulDrawable = {
    /**
     * Given the type (constructor) of a drawable, we'll mix in a combination of:
     * - initialization/disposal with the *State suffix
     * - mark* methods to be called on all drawables of nodes of this type, that set specific dirty flags
     *
     * This will allow drawables that mix in this type to do the following during an update:
     * 1. Check specific dirty flags (e.g. if the fill changed, update the fill of our SVG element).
     * 2. Call setToCleanState() once done, to clear the dirty flags.
     *
     * @param {function} drawableType - The constructor for the drawable type
     */
    mixin: function( drawableType ) {
      var proto = drawableType.prototype;

      // initializes, and resets (so we can support pooled states)
      proto.initializeState = function( renderer, instance ) {
        this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
        this.dirtyRadius = true;

        // adds fill/stroke-specific flags and state
        this.initializePaintableState( renderer, instance );

        return this; // allow for chaining
      };

      proto.disposeState = function() {
        this.disposePaintableState();
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
      };

      Paintable.PaintableStatefulDrawable.mixin( drawableType );
    }
  };

  /*---------------------------------------------------------------------------*
   * DOM rendering
   *----------------------------------------------------------------------------*/

  /**
   * A generated DOMSelfDrawable whose purpose will be drawing our Circle. One of these drawables will be created
   * for each displayed instance of a Circle.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Circle.CircleDOMDrawable = function CircleDOMDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( DOMSelfDrawable, Circle.CircleDOMDrawable, {
    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable mixin (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @private
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     */
    initialize: function( renderer, instance ) {
      this.initializeDOMSelfDrawable( renderer, instance );
      this.initializeState( renderer, instance );

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
        fillElement.style.pointerEvents = 'none';
        strokeElement.style.display = 'block';
        strokeElement.style.position = 'absolute';
        strokeElement.style.left = '0';
        strokeElement.style.top = '0';
        strokeElement.style.pointerEvents = 'none';
        fillElement.appendChild( strokeElement );
      }

      this.domElement = this.fillElement;

      scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );

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
          fillElement.style[ Features.borderRadius ] = node._radius + 'px';
        }
        if ( this.dirtyFill ) {
          fillElement.style.backgroundColor = node.getCSSFill();
        }

        if ( this.dirtyStroke ) {
          // update stroke presence
          if ( node.hasStroke() ) {
            strokeElement.style.borderStyle = 'solid';
          }
          else {
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
            strokeElement.style[ Features.borderRadius ] = ( node._radius + node.getLineWidth() / 2 ) + 'px';
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
        scenery.Util.applyPreparedTransform( this.matrix, this.fillElement, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.setToClean();
    },

    setToClean: function() {
      this.setToCleanState();

      this.cleanPaintableState();

      this.transformDirty = false;
    },

    dispose: function() {
      this.disposeState();

      // Release the DOM elements from the poolable visual state so they aren't kept in memory.
      // May not be done on platforms where we have enough memory to pool these
      if ( !keepDOMCircleElements ) {
        // clear the references
        this.fillElement = null;
        this.strokeElement = null;
        this.domElement = null;
      }
      DOMSelfDrawable.prototype.dispose.call( this );
    }
  } );
  Circle.CircleStatefulDrawable.mixin( Circle.CircleDOMDrawable );
  // This sets up CircleDOMDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Circle.CircleDOMDrawable );

  /*---------------------------------------------------------------------------*
   * SVG Rendering
   *----------------------------------------------------------------------------*/

  /**
   * A generated SVGSelfDrawable whose purpose will be drawing our Circle. One of these drawables will be created
   * for each displayed instance of a Circle.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Circle.CircleSVGDrawable = function CircleSVGDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( SVGSelfDrawable, Circle.CircleSVGDrawable, {
    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable mixin (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @private
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     */
    initialize: function( renderer, instance ) {
      this.initializeSVGSelfDrawable( renderer, instance, true, keepSVGCircleElements ); // usesPaint: true

      if ( !this.svgElement ) {
        this.svgElement = document.createElementNS( scenery.svgns, 'circle' );
      }

      return this;
    },
    updateSVGSelf: function() {
      var circle = this.svgElement;

      if ( this.dirtyRadius ) {
        circle.setAttribute( 'r', this.node._radius );
      }

      this.updateFillStrokeStyle( circle );
    }
  } );
  Circle.CircleStatefulDrawable.mixin( Circle.CircleSVGDrawable );
  // This sets up CircleSVGDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Circle.CircleSVGDrawable );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  /**
   * A generated CanvasSelfDrawable whose purpose will be drawing our Circle. One of these drawables will be created
   * for each displayed instance of a Circle.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Circle.CircleCanvasDrawable = function CircleCanvasDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( CanvasSelfDrawable, Circle.CircleCanvasDrawable, {
    /**
     * Initializes this drawable, starting its "lifetime" until it is disposed. This lifecycle can happen multiple
     * times, with instances generally created by the SelfDrawable.Poolable mixin (dirtyFromPool/createFromPool), and
     * disposal will return this drawable to the pool.
     * @private
     *
     * This acts as a pseudo-constructor that can be called multiple times, and effectively creates/resets the state
     * of the drawable to the initial state.
     *
     * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
     * @param {Instance} instance
     */
    initialize: function( renderer, instance ) {
      this.initializeCanvasSelfDrawable( renderer, instance );
      this.initializePaintableStateless( renderer, instance );
      return this;
    },

    /**
     * Paints this drawable to a Canvas (the wrapper contains both a Canvas reference and its drawing context).
     * @public
     *
     * Assumes that the Canvas's context is already in the proper local coordinate frame for the node, and that any
     * other required effects (opacity, clipping, etc.) have already been prepared.
     *
     * This is part of the CanvasSelfDrawable API required to be implemented for subtypes.
     *
     * @param {CanvasContextWrapper} wrapper - Contains the Canvas and its drawing context
     * @param {Node} node - Our node that is being drawn
     */
    paintCanvas: function( wrapper, node ) {
      var context = wrapper.context;

      context.beginPath();
      context.arc( 0, 0, node._radius, 0, Math.PI * 2, false );
      context.closePath();

      if ( node.hasFill() ) {
        node.beforeCanvasFill( wrapper ); // defined in Paintable
        context.fill();
        node.afterCanvasFill( wrapper ); // defined in Paintable
      }
      if ( node.hasStroke() ) {
        node.beforeCanvasStroke( wrapper ); // defined in Paintable
        context.stroke();
        node.afterCanvasStroke( wrapper ); // defined in Paintable
      }
    },

    // stateless dirty functions
    markDirtyRadius: function() { this.markPaintDirty(); },

    dispose: function() {
      CanvasSelfDrawable.prototype.dispose.call( this );
      this.disposePaintableStateless();
    }
  } );
  Paintable.PaintableStatelessDrawable.mixin( Circle.CircleCanvasDrawable );
  // This sets up CircleCanvasDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Circle.CircleCanvasDrawable );

  return Circle;
} );
