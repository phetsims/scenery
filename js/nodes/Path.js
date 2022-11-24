// Copyright 2013-2015, University of Colorado Boulder

/**
 * A Path draws a Shape with a specific type of fill and stroke.
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

  function Path( shape, options ) {
    // TODO: consider directly passing in a shape object (or at least handling that case)
    // NOTE: _shape can be lazily constructed, in the case of types like Rectangle where they have their own drawing code
    this._shape = null;
    this._strokedShape = null; // a stroked copy of the shape, lazily computed

    // boundsMethod determines how our (self) bounds are computed, and can particularly determine how expensive
    // to compute our bounds are if we are stroked. There are the following options:
    // 'accurate' - Always uses the most accurate way of getting bounds
    // 'unstroked' - Ignores any stroke, just gives the filled bounds.
    //               If there is a stroke, the bounds will be marked as inaccurate
    // 'tightPadding' - Pads the filled bounds by enough to cover everything except mitered joints.
    //                   If there is a stroke, the bounds wil be marked as inaccurate.
    // 'safePadding' - Pads the filled bounds by enough to cover all line joins/caps.
    // 'none' - Returns Bounds2.NOTHING. The bounds will be marked as inaccurate.
    this._boundsMethod = 'accurate'; // 'accurate', 'unstroked', 'tightPadding', 'safePadding', 'none'

    // ensure we have a parameter object
    options = options || {};

    // Used as a listener to Shapes for when they are invalidated
    this._invalidShapeListener = this.invalidateShape.bind( this );

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

  scenery.register( 'Path', Path );

  inherit( Node, Path, {
    // allow more specific path types (Rectangle, Line) to override what restrictions we have
    getPathRendererBitmask: function() {
      return Renderer.bitmaskCanvas | Renderer.bitmaskSVG;
    },

    invalidateSupportedRenderers: function() {
      this.setRendererBitmask( this.getFillRendererBitmask() & this.getStrokeRendererBitmask() & this.getPathRendererBitmask() );
    },

    // sets the shape drawn, or null to remove the shape
    setShape: function( shape ) {
      if ( this._shape !== shape ) {
        // Remove Shape invalidation listener if applicable
        if ( this._shape ) {
          this._shape.offStatic( 'invalidated', this._invalidShapeListener );
        }

        if ( typeof shape === 'string' ) {
          // be content with setShape always invalidating the shape?
          shape = new Shape( shape );
        }
        this._shape = shape;
        this.invalidateShape();

        // Add Shape invalidation listener if applicable
        if ( this._shape ) {
          this._shape.onStatic( 'invalidated', this._invalidShapeListener );
        }
      }
      return this;
    },
    set shape( value ) { this.setShape( value ); },

    getShape: function() {
      return this._shape;
    },
    get shape() { return this.getShape(); },

    getStrokedShape: function() {
      if ( !this._strokedShape ) {
        this._strokedShape = this.getShape().getStrokedShape( this._lineDrawingStyles );
      }
      return this._strokedShape;
    },

    /**
     * Invalidates the Shape stored itself. Should mainly only be called on Path itself, not subtypes like
     * Line/Rectangle/Circle/etc. once constructed.
     */
    invalidateShape: function() {
      this.invalidatePath();

      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyShape(); // subtypes of Path may not have this, but it's called during construction
      }
    },

    /**
     * Invalidates the self-bounds, that could have changed from different things.
     */
    invalidatePath: function() {
      this._strokedShape = null;

      this.invalidateSelf(); // We don't immediately compute the bounds
    },

    /**
     * @override
     *
     * @returns {boolean}
     */
    updateSelfBounds: function() {
      var selfBounds = this.hasShape() ? this.computeShapeBounds() : Bounds2.NOTHING;
      var changed = !selfBounds.equals( this._selfBounds );
      if ( changed ) {
        this._selfBounds.set( selfBounds );
      }
      return changed;
    },

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

    getBoundsMethod: function() {
      return this._boundsMethod;
    },
    get boundsMethod() { return this.getBoundsMethod(); },

    // separated out, so that we can override this with a faster version in subtypes. includes the Stroke, if any
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

    // @override
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

    // @override
    getTransformedSelfBounds: function( matrix ) {
      return ( this._stroke ? this.getStrokedShape() : this.getShape() ).getBoundsWithTransform( matrix );
    },

    // hook stroke mixin changes to invalidation
    invalidateStroke: function() {
      this.invalidatePath();
      this.trigger0( 'selfBoundsValid' ); // Stroke changing could have changed our self-bounds-validitity (unstroked/etc)
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
        path.setAttribute( 'd', `${svgPath}${platform.safari ? ' M0 0' : ''}` );
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


