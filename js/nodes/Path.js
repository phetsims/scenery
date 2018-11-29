// Copyright 2013-2016, University of Colorado Boulder

/**
 * A Path draws a Shape with a specific type of fill and stroke. Mixes in Paintable.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Bounds2 = require( 'DOT/Bounds2' );
  var extendDefined = require( 'PHET_CORE/extendDefined' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Paintable = require( 'SCENERY/nodes/Paintable' );
  var PathCanvasDrawable = require( 'SCENERY/display/drawables/PathCanvasDrawable' );
  var PathSVGDrawable = require( 'SCENERY/display/drawables/PathSVGDrawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var Shape = require( 'KITE/Shape' );

  var PATH_OPTION_KEYS = [
    'boundsMethod', // Sets how bounds are determined, see setBoundsMethod() for more documentation.
    'shape' // Sets the shape of the Path, see  setShape() for more documentation.
  ];

  var DEFAULT_OPTIONS = {
    shape: null,
    boundsMethod: 'accurate'
  };

  /**
   * Creates a Path with a given shape specifier (a Shape, a string in the SVG path format, or null to indicate no
   * shape).
   * @public
   * @constructor
   * @extends Node
   * @mixes Paintable
   *
   * Path has two additional options (above what Node provides):
   * - shape: The actual Shape (or a string representing an SVG path, or null).
   * - boundsMethod: Determines how the bounds of a shape are determined.
   *
   * @param {Shape|string|null} shape - The initial Shape to display. See setShape() for more details and documentation.
   * @param {Object} [options] - Path-specific options are documented in PATH_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  function Path( shape, options ) {
    assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
      'Extra prototype on Node options object is a code smell' );

    // @private {Shape|null} - The Shape used for displaying this Path.
    // NOTE: _shape can be lazily constructed in subtypes (may be null) if hasShape() is overridden to retun true,
    //       like in Rectangle. This is because usually the actual Shape is already implied by other parameters,
    //       so it is best to not have to compute it on changes.
    // NOTE: Please use hasShape() to determine if we are actually drawing things, as it is subtype-safe.
    this._shape = DEFAULT_OPTIONS.shape;

    // @private {Shape|null}
    // This stores a stroked copy of the Shape which is lazily computed. This can be required for computing bounds
    // of a Shape with a stroke.
    this._strokedShape = null;

    // @private {string}, one of 'accurate', 'unstroked', 'tightPadding', 'safePadding', 'none', see setBoundsMethod()
    this._boundsMethod = DEFAULT_OPTIONS.boundsMethod;

    // @private {Function}, called with no arguments, return value not checked.
    // Used as a listener to Shapes for when they are invalidated. The listeners are not added if the Shape is
    // immutable, and if the Shape becomes immutable, then the listeners are removed.
    this._invalidShapeListener = this.invalidateShape.bind( this );

    // @private {boolean} Whether our shape listener is attached to a shape.
    this._invalidShapeListenerAttached = false;

    this.initializePaintable();

    Node.call( this );

    this.invalidateSupportedRenderers();

    options = extendDefined( {
      shape: shape
    }, options );

    this.mutate( options );
  }

  scenery.register( 'Path', Path );

  inherit( Node, Path, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: PATH_OPTION_KEYS.concat( Node.prototype._mutatorKeys ),

    /**
     * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
     *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
     *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
     * @public (scenery-internal)
     * @override
     */
    drawableMarkFlags: Node.prototype.drawableMarkFlags.concat( [ 'shape' ] ),

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
     *       that the Shape would have, or call dispose() on the Path if it is not needed anymore.
     *
     * @param {Shape|string|null} shape
     * @returns {Path} - Returns 'this' reference, for chaining
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
      // By default, Canvas and SVG are accepted.
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
     * Invalidates the node's self-bounds and any other recorded metadata about the outline or bounds of the Shape.
     * @protected
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

      // Do not attach shape listeners if we are disposed
      if ( !this.isDisposed ) {
        this._shape.onStatic( 'invalidated', this._invalidShapeListener );
        this._invalidShapeListenerAttached = true;
      }
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
     * @returns {Path} - Returns 'this' reference, for chaining
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
        if ( !this.hasPaintableStroke() || this._boundsMethod === 'unstroked' ) {
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
     * Returns our safe self bounds when our rendered self is transformed by the matrix.
     * @public
     *
     * @param {Matrix3} matrix
     * @returns {Bounds2}
     */
    getTransformedSafeSelfBounds: function( matrix ) {
      return this.getTransformedSelfBounds( matrix );
    },

    /**
     * Called from (and overridden in) the Paintable trait, invalidates our current stroke, triggering recomputation of
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
     * @param {Matrix3} matrix - The transformation matrix already applied to the context.
     */
    canvasPaintSelf: function( wrapper, matrix ) {
      //TODO: Have a separate method for this, instead of touching the prototype. Can make 'this' references too easily.
      PathCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
    },

    /**
     * Creates a SVG drawable for this Path.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {SVGSelfDrawable}
     */
    createSVGDrawable: function( renderer, instance ) {
      return PathSVGDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a Canvas drawable for this Path.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {CanvasSelfDrawable}
     */
    createCanvasDrawable: function( renderer, instance ) {
      return PathCanvasDrawable.createFromPool( renderer, instance );
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
      if ( !this._stroke || !this._stroke.isPaint || !this.hasShape() ) {
        return false;
      }

      var bounds = this.computeShapeBounds();
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
     * Disposes the path, releasing shape listeners if needed (and preventing new listeners from being added).
     * @public
     * @override
     */
    dispose: function() {
      if ( this._invalidShapeListenerAttached ) {
        this.detachShapeListener();
      }

      Node.prototype.dispose.call( this );
    }
  } );

  // @public {Object} - Initial values for most Node mutator options
  Path.DEFAULT_OPTIONS = _.extend( {}, Node.DEFAULT_OPTIONS, DEFAULT_OPTIONS );

  // mix in support for fills and strokes
  Paintable.mixInto( Path );

  return Path;
} );
