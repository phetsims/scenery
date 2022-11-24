// Copyright 2013-2015, University of Colorado Boulder

/**
 * A Path draws a Shape with a specific type of fill and stroke. Mixes in Paintable.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var platform = require( 'PHET_CORE/platform' );
  var Shape = require( 'KITE/Shape' );
  var Bounds2 = require( 'DOT/Bounds2' );

  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var Paintable = require( 'SCENERY/nodes/Paintable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepSVGPathElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  /**
   * Creates a Path with a given shape specifier (a Shape, a string in the SVG path format, or null to indicate no
   * shape).
   * @constructor
   *
   * Path has two additional options (above what Node provides):
   * - shape: The actual Shape (or a string representing an SVG path, or null).
   * - boundsMethod: Determines how the bounds of a shape are determined.
   *
   * @param {Shape|string|null} shape
   * @param {Object} [options] - All options passed through to Node
   */
  function Path( shape, options ) {
    // @private {Shape|null}
    // NOTE: _shape can be lazily constructed in subtypes (may be null) if hasShape() is overridden to retun true,
    //       like in Rectangle. This is because usually the actual Shape is already implied by other parameters,
    //       so it is best to not have to compute it on changes.
    // NOTE: Please use hasShape() to determine if we are actually drawing things, as it is subtype-safe.
    this._shape = null;

    // @private {Shape|null}
    // This stores a stroked copy of the Shape which is lazily computed. This can be required for computing bounds
    // of a Shape with a stroke.
    this._strokedShape = null;

    // @private {String}, one of 'accurate', 'unstroked', 'tightPadding', 'safePadding', 'none'
    // See setBoundsMethod for details.
    this._boundsMethod = 'accurate'; // 'accurate', 'unstroked', 'tightPadding', 'safePadding', 'none'

    // If a parameter object is not provided, create an empty one
    options = options || {};

    // @private {Function}, called with no arguments, return value not checked.
    // Used as a listener to Shapes for when they are invalidated. The listeners are not added if the Shape is
    // immutable, and if the Shape becomes immutable, then the listeners are removed.
    this._invalidShapeListener = this.invalidateShape.bind( this );

    // @private {boolean} Whether our shape listener is attached to a shape.
    this._invalidShapeListenerAttached = false;

    this.initializePaintable();

    Node.call( this );
    this.invalidateSupportedRenderers();

    // Set up the boundsMethod first before setting the Shape, see https://github.com/phetsims/scenery/issues/489
    if ( options.boundsMethod ) {
      this.setBoundsMethod( options.boundsMethod );
    }
    this.setShape( shape );

    this.mutate( options );
  }

  scenery.register( 'Path', Path ); // Also mixes in Paintable.

  inherit( Node, Path, {
    /**
     * This sets the shape of the Path, which determines the shape of its appearance. It should generally not be called
     * on Path subtypes like Line, Rectangle, etc.
     * @public
     *
     * NOTE: When you create a Path with a shape in the constructor, this function will be called.
     *
     * The valid parameter types are:
     * - Shape: (from Kite), normally used.
     * - string: Uses the SVG Path format, see https://www.w3.org/TR/SVG/paths.html (the PATH part of <path d="PATH"/>).
     *           This will immediately be converted to a Shape object, and getShape() or equivalents will return the new
     *           Shape object instead of the original string.
     * - null: Indicates that there is no Shape, and nothing is drawn. Usually used as a placeholder.
     *
     * NOTE: Be aware of the potential for memory leaks. If a Shape is not marked as immutable (with makeImmutable()),
     *       Path will add a listener so that it is updated when the Shape itself changes. If there is a listener
     *       added, keeping a reference to the Shape will also keep a reference to the Path object (and thus whatever
     *       Nodes are connected to the Path). For now, set path.shape = null if you need to release the reference
     *       that the Shape would have.
     * TODO: Add a dispose() function or equivalent, which releases the listener.
     *
     * @param {Shape|string|null} shape
     */
    setShape: function( shape ) {
      assert && assert( shape === null || typeof shape === 'string' || shape instanceof Shape,
        'A path\'s shape should either be null, a string, or a Shape' );

      if ( this._shape !== shape ) {
        // Remove Shape invalidation listener if applicable
        if ( this._invalidShapeListenerAttached ) {
          this.detachShapeListener();
        }

        if ( typeof shape === 'string' ) {
          // be content with setShape always invalidating the shape?
          shape = new Shape( shape );
        }
        this._shape = shape;
        this.invalidateShape();

        // Add Shape invalidation listener if applicable
        if ( this._shape && !this._shape.isImmutable() ) {
          this.attachShapeListener();
        }
      }
      return this;
    },
    set shape( value ) { this.setShape( value ); },

    /**
     * Returns the shape that was set for this Path (or for subtypes like Line and Rectangle, will return an immutable
     * Shape that is equivalent in appearance).
     * @public
     *
     * It is best to generally assume modifications to the Shape returned is not supported. If there is no shape
     * currently, null will be returned.
     *
     * @returns {Shape|null}
     */
    getShape: function() {
      return this._shape;
    },
    get shape() { return this.getShape(); },

    /**
     * Returns a lazily-created Shape that has the appearance of the Path's shape but stroked using the current
     * stroke style of the Path.
     * @public
     *
     * NOTE: It is invalid to call this on a Path that does not currently have a Shape (usually a Path where
     *       the shape is set to null).
     *
     * @returns {Shape}
     */
    getStrokedShape: function() {
      assert && assert( this.hasShape(), 'We cannot stroke a non-existing shape' );

      // Lazily compute the stroked shape. It should be set to null when we need to recompute it
      if ( !this._strokedShape ) {
        this._strokedShape = this.getShape().getStrokedShape( this._lineDrawingStyles );
      }

      return this._strokedShape;
    },

    /**
     * Returns a bitmask representing the supported renderers for the current configuration of the Path or subtype.
     * @protected
     *
     * Should be overridden by subtypes to either extend or restrict renderers, depending on what renderers are
     * supported.
     *
     * @returns {number} - A bitmask that includes supported renderers, see Renderer for details.
     */
    getPathRendererBitmask: function() {
      return Renderer.bitmaskCanvas | Renderer.bitmaskSVG;
    },

    /**
     * Triggers a check and update for what renderers the current configuration of this Path or subtype supports.
     * This should be called whenever something that could potentially change supported renderers happen (which can
     * be the shape, properties of the strokes or fills, etc.)
     * @public
     */
    invalidateSupportedRenderers: function() {
      this.setRendererBitmask( this.getFillRendererBitmask() & this.getStrokeRendererBitmask() & this.getPathRendererBitmask() );
    },

    /**
     * Notifies the Path that the Shape has changed (either the Shape itself has be mutated, a new Shape has been
     * provided).
     * @private
     *
     * NOTE: This should not be called on subtypes of Path after they have been constructed, like Line, Rectangle, etc.
     */
    invalidateShape: function() {
      this.invalidatePath();

      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyShape(); // subtypes of Path may not have this, but it's called during construction
      }

      // Disconnect our Shape listener if our Shape has become immutable.
      // see https://github.com/phetsims/sun/issues/270#issuecomment-250266174
      if ( this._invalidShapeListenerAttached && this._shape && this._shape.isImmutable() ) {
        this.detachShapeListener();
      }
    },

    /**
     * Invalidates the node's self-bounds and any other recorded metadata about the outline or bound sof the Shape.
     * @private
     *
     * This is meant to be used for all Path subtypes (unlike invalidateShape).
     */
    invalidatePath: function() {
      this._strokedShape = null;

      this.invalidateSelf(); // We don't immediately compute the bounds
    },

    /**
     * Attaches a listener to our Shape that will be called whenever the Shape changes.
     * @private
     */
    attachShapeListener: function() {
      assert && assert( !this._invalidShapeListenerAttached, 'We do not want to have two listeners attached!' );

      this._shape.onStatic( 'invalidated', this._invalidShapeListener );
      this._invalidShapeListenerAttached = true;
    },

    /**
     * Detaches a previously-attached listener added to our Shape (see attachShapeListener).
     * @private
     */
    detachShapeListener: function() {
      assert && assert( this._invalidShapeListenerAttached, 'We cannot detach an unattached listener' );

      this._shape.offStatic( 'invalidated', this._invalidShapeListener );
      this._invalidShapeListenerAttached = false;
    },

    /**
     * Computes a more efficient selfBounds for our Path.
     * @protected
     * @override
     *
     * @returns {boolean} - Whether the self bounds changed.
     */
    updateSelfBounds: function() {
      var selfBounds = this.hasShape() ? this.computeShapeBounds() : Bounds2.NOTHING;
      var changed = !selfBounds.equals( this._selfBounds );
      if ( changed ) {
        this._selfBounds.set( selfBounds );
      }
      return changed;
    },

    /**
     * Sets the bounds method for the Path. This determines how our (self) bounds are computed, and can particularly
     * determine how expensive to compute our bounds are if we have a stroke.
     * @public
     *
     * There are the following options:
     * - 'accurate' - Always uses the most accurate way of getting bounds. Computes the exact stroked bounds.
     * - 'unstroked' - Ignores any stroke, just gives the filled bounds.
     *                 If there is a stroke, the bounds will be marked as inaccurate
     * - 'tightPadding' - Pads the filled bounds by enough to cover everything except mitered joints.
     *                     If there is a stroke, the bounds wil be marked as inaccurate.
     * - 'safePadding' - Pads the filled bounds by enough to cover all line joins/caps.
     * - 'none' - Returns Bounds2.NOTHING. The bounds will be marked as inaccurate.
     *
     * @param {string} boundsMethod - one of 'accurate', 'unstroked', 'tightPadding', 'safePadding' or 'none'
     */
    setBoundsMethod: function( boundsMethod ) {
      assert && assert( boundsMethod === 'accurate' ||
                        boundsMethod === 'unstroked' ||
                        boundsMethod === 'tightPadding' ||
                        boundsMethod === 'safePadding' ||
                        boundsMethod === 'none' );
      if ( this._boundsMethod !== boundsMethod ) {
        this._boundsMethod = boundsMethod;
        this.invalidatePath();

        this.trigger0( 'boundsMethod' );

        this.trigger0( 'selfBoundsValid' ); // whether our self bounds are valid may have changed
      }
      return this;
    },
    set boundsMethod( value ) { return this.setBoundsMethod( value ); },

    /**
     * Returns the curent bounds method. See setBoundsMethod for details.
     * @public
     *
     * @returns {string}
     */
    getBoundsMethod: function() {
      return this._boundsMethod;
    },
    get boundsMethod() { return this.getBoundsMethod(); },

    /**
     * Computes the bounds of the Path (or subtype when overridden). Meant to be overridden in subtypes for more
     * efficient bounds computations (but this will work as a fallback). Includes the stroked region if there is a
     * stroke applied to the Path.
     * @public
     *
     * @returns {Bounds2}
     */
    computeShapeBounds: function() {
      // boundsMethod: 'none' will return no bounds
      if ( this._boundsMethod === 'none' ) {
        return Bounds2.NOTHING;
      }
      else {
        // boundsMethod: 'unstroked', or anything without a stroke will then just use the normal shape bounds
        if ( !this.hasStroke() || this.getLineWidth() === 0 || this._boundsMethod === 'unstroked' ) {
          return this.getShape().bounds;
        }
        else {
          // 'accurate' will always require computing the full stroked shape, and taking its bounds
          if ( this._boundsMethod === 'accurate' ) {
            return this.getShape().getStrokedBounds( this.getLineStyles() );
          }
          // Otherwise we compute bounds based on 'tightPadding' and 'safePadding', the one difference being that
          // 'safePadding' will include whatever bounds necessary to include miters. Square line-cap requires a
          // slightly extended bounds in either case.
          else {
            var factor;
            // If miterLength (inside corner to outside corner) exceeds miterLimit * strokeWidth, it will get turned to
            // a bevel, so our factor will be based just on the miterLimit.
            if ( this._boundsMethod === 'safePadding' && this.getLineJoin() === 'miter' ) {
              factor = this.getMiterLimit();
            }
            else if ( this.getLineCap() === 'square' ) {
              factor = Math.SQRT2;
            }
            else {
              factor = 1;
            }
            return this.getShape().bounds.dilated( factor * this.getLineWidth() / 2 );
          }
        }
      }
    },

    /**
     * Whether this Node's selfBounds are considered to be valid (always containing the displayed self content
     * of this node). Meant to be overridden in subtypes when this can change (e.g. Text).
     * @public
     * @override
     *
     * If this value would potentially change, please trigger the event 'selfBoundsValid'.
     *
     * @returns {boolean}
     */
    areSelfBoundsValid: function() {
      if ( this._boundsMethod === 'accurate' || this._boundsMethod === 'safePadding' ) {
        return true;
      }
      else if ( this._boundsMethod === 'none' ) {
        return false;
      }
      else {
        return !this.hasStroke(); // 'tightPadding' and 'unstroked' options
      }
    },

    /**
     * Returns our self bounds when our rendered self is transformed by the matrix.
     * @public
     * @override
     *
     * @param {Matrix3} matrix
     * @returns {Bounds2}
     */
    getTransformedSelfBounds: function( matrix ) {
      return ( this._stroke ? this.getStrokedShape() : this.getShape() ).getBoundsWithTransform( matrix );
    },

    /**
     * Called from (and overridden in) the Paintable mixin, invalidates our current stroke, triggering recomputation of
     * anything that depended on the old stroke's value.
     * @protected (scenery-internal)
     */
    invalidateStroke: function() {
      this.invalidatePath();
      this.trigger0( 'selfBoundsValid' ); // Stroke changing could have changed our self-bounds-validitity (unstroked/etc)
    },

    /**
     * Returns whether this Path has an associated Shape (instead of no shape, represented by null)
     * @public
     *
     * @returns {boolean}
     */
    hasShape: function() {
      return !!this._shape;
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
      Path.PathCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    /**
     * Creates a SVG drawable for this Path.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {SVGSelfDrawable}
     */
    createSVGDrawable: function( renderer, instance ) {
      return Path.PathSVGDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a Canvas drawable for this Path.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {CanvasSelfDrawable}
     */
    createCanvasDrawable: function( renderer, instance ) {
      return Path.PathCanvasDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a WebGL drawable for this Path.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {WebGLSelfDrawable}
     */
    createWebGLDrawable: function( renderer, instance ) {
      return Path.PathWebGLDrawable.createFromPool( renderer, instance );
    },

    /**
     * Whether this Node itself is painted (displays something itself).
     * @public
     * @override
     *
     * @returns {boolean}
     */
    isPainted: function() {
      // Always true for Path nodes
      return true;
    },

    /**
     * Computes whether the provided point is "inside" (contained) in this Path's self content, or "outside".
     * @protected
     * @override
     *
     * @param {Vector2} point - Considered to be in the local coordinate frame
     * @returns {boolean}
     */
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

    /**
     * Returns whether this Path's selfBounds is intersected by the specified bounds.
     * @public
     *
     * @param {Bounds2} bounds - Bounds to test, assumed to be in the local coordinate frame.
     * @returns {boolean}
     */
    intersectsBoundsSelf: function( bounds ) {
      // TODO: should a shape's stroke be included?
      return this.hasShape() ? this._shape.intersectsBounds( bounds ) : false;
    },

    /**
     * Returns whether we need to apply a transform workaround for https://github.com/phetsims/scenery/issues/196, which
     * only applies when we have a pattern or gradient (e.g. subtypes of Paint).
     * @private
     *
     * @returns {boolean}
     */
    requiresSVGBoundsWorkaround: function() {
      if ( !this._stroke || !this._stroke.getSVGDefinition || !this.hasShape() ) {
        return false;
      }

      var bounds = this.computeShapeBounds( false ); // without stroke
      return bounds.x * bounds.y === 0; // at least one of them was zero, so the bounding box has no area
    },

    /**
     * Override for extra information in the debugging output (from Display.getDebugHTML()).
     * @protected (scenery-internal)
     * @override
     *
     * @returns {string}
     */
    getDebugHTMLExtras: function() {
      return this._shape ? ' (<span style="color: #88f" onclick="window.open( \'data:text/plain;charset=utf-8,\' + encodeURIComponent( \'' + this._shape.getSVGPath() + '\' ) );">path</span>)' : '';
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
      return 'new scenery.Path( ' + ( this._shape ? this._shape.toString() : this._shape ) + ', {' + propLines + '} )';
    },

    /**
     * Returns the property object string for use with toString().
     * @protected (scenery-internal)
     * @override
     *
     * @param {string} spaces - Whitespace to add
     * @param {boolean} [includeChildren]
     */
    getPropString: function( spaces, includeChildren ) {
      var result = Node.prototype.getPropString.call( this, spaces, includeChildren );
      result = this.appendFillablePropString( spaces, result );
      result = this.appendStrokablePropString( spaces, result );
      return result;
    }
  } );

  Path.prototype._mutatorKeys = [ 'boundsMethod', 'shape' ].concat( Node.prototype._mutatorKeys );

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
      proto.initializeState = function( renderer, instance ) {
        this.paintDirty = true; // flag that is marked if ANY "paint" dirty flag is set (basically everything except for transforms, so we can accelerated the transform-only case)
        this.dirtyShape = true;

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

      proto.markDirtyShape = function() {
        this.dirtyShape = true;
        this.markPaintDirty();
      };

      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyShape = false;
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

        // We'll conditionally add another M0 0 to the end of the path if we're on Safari, we're running into a bug in
        // https://github.com/phetsims/gravity-and-orbits/issues/472 (debugged in
        // https://github.com/phetsims/geometric-optics-basics/issues/31) where we're getting artifacts.
        path.setAttribute( 'd', svgPath + ( platform.safari ? ' M0 0' : '' ) );
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
      this.initializeCanvasSelfDrawable( renderer, instance );
      this.initializePaintableStateless( renderer, instance );
      return this;
    },

    paintCanvas: function( wrapper, node ) {
      var context = wrapper.context;

      if ( node.hasShape() ) {
        // TODO: fill/stroke delay optimizations?
        context.beginPath();
        node._shape.writeToContext( context );

        if ( node.hasFill() ) {
          node.beforeCanvasFill( wrapper ); // defined in Paintable
          context.fill();
          node.afterCanvasFill( wrapper ); // defined in Paintable
        }

        // Do not render strokes in Canvas if the lineWidth is 0, see https://github.com/phetsims/scenery/issues/523
        if ( node.hasStroke() && node.getLineWidth() > 0 ) {
          node.beforeCanvasStroke( wrapper ); // defined in Paintable
          context.stroke();
          node.afterCanvasStroke( wrapper ); // defined in Paintable
        }
      }
    },

    // stateless dirty functions
    markDirtyShape: function() { this.markPaintDirty(); },

    dispose: function() {
      CanvasSelfDrawable.prototype.dispose.call( this );
      this.disposePaintableStateless();
    }
  } );
  Paintable.PaintableStatelessDrawable.mixin( Path.PathCanvasDrawable );
  SelfDrawable.Poolable.mixin( Path.PathCanvasDrawable );

  return Path;
} );


