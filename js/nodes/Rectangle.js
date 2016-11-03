// Copyright 2013-2015, University of Colorado Boulder

/**
 * A rectangular node that inherits Path, and allows for optimized drawing,
 * and improved rectangle handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Vector2 = require( 'DOT/Vector2' );
  var Dimension2 = require( 'DOT/Dimension2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Property = require( 'AXON/Property' );
  var Features = require( 'SCENERY/util/Features' );
  var Paintable = require( 'SCENERY/nodes/Paintable' );
  var DOMSelfDrawable = require( 'SCENERY/display/DOMSelfDrawable' );
  var SVGSelfDrawable = require( 'SCENERY/display/SVGSelfDrawable' );
  var CanvasSelfDrawable = require( 'SCENERY/display/CanvasSelfDrawable' );
  var WebGLSelfDrawable = require( 'SCENERY/display/WebGLSelfDrawable' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var Color = require( 'SCENERY/util/Color' );

  // TODO: change this based on memory and performance characteristics of the platform
  var keepDOMRectangleElements = true; // whether we should pool DOM elements for the DOM rendering states, or whether we should free them when possible for memory
  var keepSVGRectangleElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

  // scratch matrix used in DOM rendering
  var scratchMatrix = Matrix3.dirtyFromPool();

  /**
   * @constructor
   * @public
   *
   * Possible constructor signatures
   * new Rectangle( x, y, width, height, cornerXRadius, cornerYRadius, [options] )
   * new Rectangle( x, y, width, height, [options] )
   * new Rectangle( [options] )
   * new Rectangle( bounds2, [options] )
   * new Rectangle( bounds2, cornerXRadius, cornerYRadius, [options] )
   *
   * Available parameters to the various constructor options:
   * @param {number} x - x-position of the upper-left corner (left bound)
   * @param {number} y - y-position of the upper-left corner (top bound)
   * @param {number} width - width of the rectangle to the right of the upper-left corner, required to be >= 0
   * @param {number} height - height of the rectangle below the upper-left corner, required to be >= 0
   * @param {number} cornerXRadius - positive vertical radius (width) of the rounded corner, or 0 to indicate the corner should be sharp
   * @param {number} cornerYRadius - positive horizontal radius (height) of the rounded corner, or 0 to indicate the corner should be sharp
   * @param {Object} [options] - Options object for Scenery
   *
   * Current available options for the options object (custom for Rectangle, not Path or Node):
   * rectX - Left edge of the rectangle in the local coordinate frame
   * rectY - Top edge of the rectangle in the local coordinate frame
   * rectWidth - Width of the rectangle in the local coordinate frame
   * rectHeight - Height of the rectangle in the local coordinate frame
   * cornerXRadius - The x-axis radius for elliptical/circular rounded corners.
   * cornerYRadius - The y-axis radius for elliptical/circular rounded corners.
   * cornerRadius - Sets both "X" and "Y" corner radii above.
   *
   * NOTE: the X and Y corner radii need to both be greater than zero for rounded corners to appear. If they have the
   * same non-zero value, circular rounded corners will be used.
   */
  function Rectangle( x, y, width, height, cornerXRadius, cornerYRadius, options ) {
    if ( typeof x === 'object' ) {
      if ( x instanceof Bounds2 ) {
        // allow new Rectangle( bounds2, { ... } ) or new Rectangle( bounds2, cornerXRadius, cornerYRadius, options )
        this._rectX = x.minX;
        this._rectY = x.minY;
        this._rectWidth = x.width;
        this._rectHeight = x.height;
        if ( arguments.length < 3 ) {
          // Rectangle( bounds2, { ... } )
          options = y;
          this._cornerXRadius = 0;
          this._cornerYRadius = 0;
        }
        else {
          // Rectangle( bounds2, cornerXRadius, cornerYRadius, { ... } )
          options = height;
          this._cornerXRadius = y;
          this._cornerYRadius = width;
        }
      }
      else {
        // allow new Rectangle( { rectX: x, rectY: y, rectWidth: width, rectHeight: height, ... } )
        // the mutators will call invalidateRectangle() and properly set the shape
        options = x;
        this._rectX = options.rectX || 0;
        this._rectY = options.rectY || 0;
        this._rectWidth = options.rectWidth;
        this._rectHeight = options.rectHeight;
        this._cornerXRadius = options.cornerXRadius || 0;
        this._cornerYRadius = options.cornerYRadius || 0;
      }
    }
    else if ( arguments.length < 6 ) {
      // new Rectangle( x, y, width, height, [options] )
      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._cornerXRadius = 0;
      this._cornerYRadius = 0;

      // ensure we have a parameter object
      options = cornerXRadius || {};

    }
    else {
      // normal case with args (including cornerXRadius / cornerYRadius)
      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._cornerXRadius = cornerXRadius;
      this._cornerYRadius = cornerYRadius;

      // ensure we have a parameter object
      options = options || {};

    }
    // fallback for non-canvas or non-svg rendering, and for proper bounds computation

    Path.call( this, null, options );
  }

  scenery.register( 'Rectangle', Rectangle );

  inherit( Path, Rectangle, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: [ 'rectX', 'rectY', 'rectWidth', 'rectHeight',
                    'cornerRadius', 'cornerXRadius', 'cornerYRadius' ].concat( Path.prototype._mutatorKeys ),

    /**
     * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
     *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
     *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
     * @public (scenery-internal)
     * @override
     */
    drawableMarkFlags: Path.prototype.drawableMarkFlags.concat( [ 'x', 'y', 'width', 'height', 'cornerXRadius', 'cornerYRadius' ] ).filter( function( flag ) {
      // We don't want the shape flag, as that won't be called for Path subtypes.
      return flag !== 'shape';
    } ),

    getMaximumArcSize: function() {
      return Math.min( this._rectWidth / 2, this._rectHeight / 2 );
    },

    /**
     * Determines the default allowed renderers (returned via the Renderer bitmask) that are allowed, given the
     * current stroke options.
     * @public (scenery-internal)
     * @override
     *
     * We can support the DOM renderer if there is a solid-styled stroke with non-bevel line joins
     * (which otherwise wouldn't be supported).
     *
     * @returns {number} - Renderer bitmask, see Renderer for details
     */
    getStrokeRendererBitmask: function() {
      var bitmask = Path.prototype.getStrokeRendererBitmask.call( this );
      // DOM stroke handling doesn't YET support gradients, patterns, or dashes (with the current implementation, it shouldn't be too hard)
      if ( this.hasStroke() && !this.getStroke().isGradient && !this.getStroke().isPattern && !this.hasLineDash() ) {
        // we can't support the bevel line-join with our current DOM rectangle display
        if ( this.getLineJoin() === 'miter' || ( this.getLineJoin() === 'round' && Features.borderRadius ) ) {
          bitmask |= Renderer.bitmaskDOM;
        }
      }

      if ( !this.hasStroke() ) {
        bitmask |= Renderer.bitmaskWebGL;
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
      var bitmask = Renderer.bitmaskCanvas | Renderer.bitmaskSVG;

      var maximumArcSize = this.getMaximumArcSize();

      // If the top/bottom or left/right strokes touch and overlap in the middle (small rectangle, big stroke), our DOM method won't work.
      // Additionally, if we're handling rounded rectangles or a stroke with lineJoin 'round', we'll need borderRadius
      // We also require for DOM that if it's a rounded rectangle, it's rounded with circular arcs (for now, could potentially do a transform trick!)
      if ( ( !this.hasStroke() || ( this.getLineWidth() <= this._rectHeight && this.getLineWidth() <= this._rectWidth ) ) &&
           ( !this.isRounded() || ( Features.borderRadius && this._cornerXRadius === this._cornerYRadius ) ) &&
           this._cornerYRadius <= maximumArcSize && this._cornerXRadius <= maximumArcSize ) {
        bitmask |= Renderer.bitmaskDOM;
      }

      // TODO: why check here, if we also check in the 'stroke' portion?
      if ( !this.hasStroke() && !this.isRounded() ) {
        bitmask |= Renderer.bitmaskWebGL;
      }

      return bitmask;
    },

    setRect: function( x, y, width, height, arcWidth, arcHeight ) {
      assert && assert( x !== undefined && y !== undefined && width !== undefined && height !== undefined, 'x/y/width/height need to be defined' );

      // for now, check whether this is needed
      // TODO: note that this could decrease performance? Remove if this is a bottleneck
      if ( this._rectX === x &&
           this._rectY === y &&
           this._rectWidth === width &&
           this._rectHeight === height &&
           this._cornerXRadius === arcWidth &&
           this._cornerYRadius === arcHeight ) {
        return;
      }

      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._cornerXRadius = arcWidth || 0;
      this._cornerYRadius = arcHeight || 0;

      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyRectangle();
      }
      this.invalidateRectangle();
    },

    // sets the Rectangle's x/y/width/height from the {Bounds2} bounds passed in.
    setRectBounds: function( bounds ) {
      assert && assert( bounds instanceof Bounds2 );

      this.setRect( bounds.x, bounds.y, bounds.width, bounds.height );
    },
    set rectBounds( value ) { this.setRectBounds( value ); },

    // gets a {Bounds2} from the Rectangle's x/y/width/height
    getRectBounds: function() {
      return Bounds2.rect( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
    },
    get rectBounds() { return this.getRectBounds(); },

    // sets the Rectangle's width/height from the {Dimension2} size passed in.
    setRectSize: function( size ) {
      assert && assert( size instanceof Dimension2 );

      this.setRectWidth( size.width );
      this.setRectHeight( size.height );
    },
    set rectSize( value ) { this.setRectSize( value ); },

    // gets a {Dimension2} from the Rectangle's width/height
    getRectSize: function() {
      return new Dimension2( this._rectWidth, this._rectHeight );
    },
    get rectSize() { return this.getRectSize(); },

    // sets the width of the rectangle while keeping its right edge (x + width) in the same position
    setRectWidthFromRight: function( width ) {
      assert && assert( typeof width === 'number' );

      if ( this._rectWidth !== width ) {
        var right = this._rectX + this._rectWidth;
        this.setRectWidth( width );
        this.setRectX( right - width );
      }
    },
    set rectWidthFromRight( value ) { this.setRectWidthFromRight( value ); },
    get rectWidthFromRight() { return this.getRectWidth(); }, // because JSHint complains

    // sets the height of the rectangle while keeping its bottom edge (y + height) in the same position
    setRectHeightFromBottom: function( height ) {
      assert && assert( typeof height === 'number' );

      if ( this._rectHeight !== height ) {
        var bottom = this._rectY + this._rectHeight;
        this.setRectHeight( height );
        this.setRectY( bottom - height );
      }
    },
    set rectHeightFromBottom( value ) { this.setRectHeightFromBottom( value ); },
    get rectHeightFromBottom() { return this.getRectHeight(); }, // because JSHint complains

    /**
     * Returns whether this rectangle has any rounding applied at its corners. If either the x or y corner radius is 0,
     * then there is no rounding applied.
     * @public
     *
     * @returns {boolean}
     */
    isRounded: function() {
      return this._cornerXRadius !== 0 && this._cornerYRadius !== 0;
    },

    /**
     * Computes the bounds of the Rectangle, including any applied stroke. Overridden for efficiency.
     * @public
     * @override
     *
     * @returns {Bounds2}
     */
    computeShapeBounds: function() {
      var bounds = new Bounds2( this._rectX, this._rectY, this._rectX + this._rectWidth, this._rectY + this._rectHeight );
      if ( this._stroke ) {
        // since we are axis-aligned, any stroke will expand our bounds by a guaranteed set amount
        bounds = bounds.dilated( this.getLineWidth() / 2 );
      }
      return bounds;
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
      return this.selfBounds.transformed( matrix );
    },

    createRectangleShape: function() {
      if ( this.isRounded() ) {
        // copy border-radius CSS behavior in Chrome, where the arcs won't intersect, in cases where the arc segments at full size would intersect each other
        var maximumArcSize = Math.min( this._rectWidth / 2, this._rectHeight / 2 );
        return Shape.roundRectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight,
          Math.min( maximumArcSize, this._cornerXRadius ), Math.min( maximumArcSize, this._cornerYRadius ) ).makeImmutable();
      }
      else {
        return Shape.rectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight ).makeImmutable();
      }
    },

    invalidateRectangle: function() {
      assert && assert( isFinite( this._rectX ), 'A rectangle needs to have a finite x (' + this._rectX + ')' );
      assert && assert( isFinite( this._rectY ), 'A rectangle needs to have a finite y (' + this._rectY + ')' );
      assert && assert( this._rectWidth >= 0 && isFinite( this._rectWidth ),
        'A rectangle needs to have a non-negative finite width (' + this._rectWidth + ')' );
      assert && assert( this._rectHeight >= 0 && isFinite( this._rectHeight ),
        'A rectangle needs to have a non-negative finite height (' + this._rectHeight + ')' );
      assert && assert( this._cornerXRadius >= 0 && isFinite( this._cornerXRadius ),
        'A rectangle needs to have a non-negative finite arcWidth (' + this._cornerXRadius + ')' );
      assert && assert( this._cornerYRadius >= 0 && isFinite( this._cornerYRadius ),
        'A rectangle needs to have a non-negative finite arcHeight (' + this._cornerYRadius + ')' );
      // assert && assert( !this.isRounded() || ( this._rectWidth >= this._cornerXRadius * 2 && this._rectHeight >= this._cornerYRadius * 2 ),
      //                                 'The rounded sections of the rectangle should not intersect (the length of the straight sections shouldn\'t be negative' );

      // sets our 'cache' to null, so we don't always have to recompute our shape
      this._shape = null;

      // should invalidate the path and ensure a redraw
      this.invalidatePath();

      // since we changed the rectangle arc width/height, it could make DOM work or not
      this.invalidateSupportedRenderers();
    },

    /**
     * Computes whether the provided point is "inside" (contained) in this Rectangle's self content, or "outside".
     * @protected
     * @override
     *
     * Handles axis-aligned optionally-rounded rectangles, although can only do optimized computation if it isn't
     * rounded. If it IS rounded, we check if a corner computation is needed (usually isn't), and only need to check
     * one corner for that test.
     *
     * @param {Vector2} point - Considered to be in the local coordinate frame
     * @returns {boolean}
     */
    containsPointSelf: function( point ) {
      var x = this._rectX;
      var y = this._rectY;
      var width = this._rectWidth;
      var height = this._rectHeight;
      var arcWidth = this._cornerXRadius;
      var arcHeight = this._cornerYRadius;
      var halfLine = this.getLineWidth() / 2;

      var result = true;
      if ( this._strokePickable ) {
        // test the outer boundary if we are stroke-pickable (if also fill-pickable, this is the only test we need)
        var rounded = this.isRounded();
        if ( !rounded && this.getLineJoin() === 'bevel' ) {
          // fall-back for bevel
          return Path.prototype.containsPointSelf.call( this, point );
        }
        var miter = this.getLineJoin() === 'miter' && !rounded;
        result = result && Rectangle.intersects( x - halfLine, y - halfLine,
            width + 2 * halfLine, height + 2 * halfLine,
            miter ? 0 : ( arcWidth + halfLine ), miter ? 0 : ( arcHeight + halfLine ),
            point );
      }

      if ( this._fillPickable ) {
        if ( this._strokePickable ) {
          return result;
        }
        else {
          return Rectangle.intersects( x, y, width, height, arcWidth, arcHeight, point );
        }
      }
      else if ( this._strokePickable ) {
        return result && !Rectangle.intersects( x + halfLine, y + halfLine,
            width - 2 * halfLine, height - 2 * halfLine,
            arcWidth - halfLine, arcHeight - halfLine,
            point );
      }
      else {
        return false; // either fill nor stroke is pickable
      }
    },

    /**
     * Returns whether this Rectangle's selfBounds is intersected by the specified bounds.
     * @public
     *
     * @param {Bounds2} bounds - Bounds to test, assumed to be in the local coordinate frame.
     * @returns {boolean}
     */
    intersectsBoundsSelf: function( bounds ) {
      return !this.computeShapeBounds().intersection( bounds ).isEmpty();
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
      Rectangle.RectangleCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    /**
     * Creates a DOM drawable for this Rectangle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {DOMSelfDrawable}
     */
    createDOMDrawable: function( renderer, instance ) {
      return Rectangle.RectangleDOMDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a SVG drawable for this Rectangle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {SVGSelfDrawable}
     */
    createSVGDrawable: function( renderer, instance ) {
      return Rectangle.RectangleSVGDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a Canvas drawable for this Rectangle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {CanvasSelfDrawable}
     */
    createCanvasDrawable: function( renderer, instance ) {
      return Rectangle.RectangleCanvasDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a WebGL drawable for this Rectangle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @returns {WebGLSelfDrawable}
     */
    createWebGLDrawable: function( renderer, instance ) {
      return Rectangle.RectangleWebGLDrawable.createFromPool( renderer, instance );
    },

    /*---------------------------------------------------------------------------*
     * Miscellaneous
     *----------------------------------------------------------------------------*/

    /**
     * Returns a string containing constructor information for Node.string().
     * @protected
     * @override
     *
     * @param {string} propLines - A string representing the options properties that need to be set.
     * @returns {string}
     */
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Rectangle( ' +
             this._rectX + ', ' + this._rectY + ', ' +
             this._rectWidth + ', ' + this._rectHeight + ', ' +
             this._cornerXRadius + ', ' + this._cornerYRadius +
             ', {' + propLines + '} )';
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
        throw new Error( 'Cannot set the shape of a scenery.Rectangle to something non-null' );
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
        this._shape = this.createRectangleShape();
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
      return true;
    },

    getCornerRadius: function() {
      assert && assert( this._cornerXRadius === this._cornerYRadius,
        'getCornerRadius() invalid if x/y radii are different' );

      return this._cornerXRadius;
    },
    get cornerRadius() { return this.getCornerRadius(); },

    setCornerRadius: function( cornerRadius ) {
      this.setCornerXRadius( cornerRadius );
      this.setCornerYRadius( cornerRadius );
      return this;
    },
    set cornerRadius( value ) { this.setCornerRadius( value ); }
  } );

  /*---------------------------------------------------------------------------*
   * Other Rectangle properties and ES5
   *----------------------------------------------------------------------------*/

  function addRectProp( name, setGetCapitalized, eventName ) {
    var getName = 'get' + setGetCapitalized;
    var setName = 'set' + setGetCapitalized;
    var privateName = '_' + name;
    var dirtyMethodName = 'markDirty' + eventName;

    Rectangle.prototype[ getName ] = function() {
      return this[ privateName ];
    };

    Rectangle.prototype[ setName ] = function( value ) {
      if ( this[ privateName ] !== value ) {
        this[ privateName ] = value;
        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          var state = this._drawables[ i ];
          state[ dirtyMethodName ]();
          state.markPaintDirty();
        }
        this.invalidateRectangle();
      }
      return this;
    };

    Object.defineProperty( Rectangle.prototype, name, {
      set: Rectangle.prototype[ setName ],
      get: Rectangle.prototype[ getName ]
    } );
  }

  addRectProp( 'rectX', 'RectX', 'X' );
  addRectProp( 'rectY', 'RectY', 'Y' );
  addRectProp( 'rectWidth', 'RectWidth', 'Width' );
  addRectProp( 'rectHeight', 'RectHeight', 'Height' );
  addRectProp( 'cornerXRadius', 'CornerXRadius', 'CornerXRadius' );
  addRectProp( 'cornerYRadius', 'CornerYRadius', 'CornerYRadius' );

  Rectangle.intersects = function( x, y, width, height, arcWidth, arcHeight, point ) {
    var result = point.x >= x &&
                 point.x <= x + width &&
                 point.y >= y &&
                 point.y <= y + height;

    if ( !result || arcWidth <= 0 || arcHeight <= 0 ) {
      return result;
    }

    // copy border-radius CSS behavior in Chrome, where the arcs won't intersect, in cases where the arc segments at full size would intersect each other
    var maximumArcSize = Math.min( width / 2, height / 2 );
    arcWidth = Math.min( maximumArcSize, arcWidth );
    arcHeight = Math.min( maximumArcSize, arcHeight );

    // we are rounded and inside the logical rectangle (if it didn't have rounded corners)

    // closest corner arc's center (we assume the rounded rectangle's arcs are 90 degrees fully, and don't intersect)
    var closestCornerX;
    var closestCornerY;
    var guaranteedInside = false;

    // if we are to the inside of the closest corner arc's center, we are guaranteed to be in the rounded rectangle (guaranteedInside)
    if ( point.x < x + width / 2 ) {
      closestCornerX = x + arcWidth;
      guaranteedInside = guaranteedInside || point.x >= closestCornerX;
    }
    else {
      closestCornerX = x + width - arcWidth;
      guaranteedInside = guaranteedInside || point.x <= closestCornerX;
    }
    if ( guaranteedInside ) { return true; }

    if ( point.y < y + height / 2 ) {
      closestCornerY = y + arcHeight;
      guaranteedInside = guaranteedInside || point.y >= closestCornerY;
    }
    else {
      closestCornerY = y + height - arcHeight;
      guaranteedInside = guaranteedInside || point.y <= closestCornerY;
    }
    if ( guaranteedInside ) { return true; }

    // we are now in the rectangular region between the logical corner and the center of the closest corner's arc.

    // offset from the closest corner's arc center
    var offsetX = point.x - closestCornerX;
    var offsetY = point.y - closestCornerY;

    // normalize the coordinates so now we are dealing with a unit circle
    // (technically arc, but we are guaranteed to be in the area covered by the arc, so we just consider the circle)
    // NOTE: we are rounded, so both arcWidth and arcHeight are non-zero (this is well defined)
    offsetX /= arcWidth;
    offsetY /= arcHeight;

    offsetX *= offsetX;
    offsetY *= offsetY;
    return offsetX + offsetY <= 1; // return whether we are in the rounded corner. see the formula for an ellipse
  };

  /**
   * Creates a rectangle with the specified x/y/width/height.
   * @public
   *
   * @param {x} number
   * @param {y} number
   * @param {width} number
   * @param {height} number
   * @param {Object} [options]
   * @returns {Rectangle}
   */
  Rectangle.rect = function( x, y, width, height, options ) {
    return new Rectangle( x, y, width, height, 0, 0, options );
  };

  /**
   * Creates a rounded rectangle with the specified x/y/width/height/arcWidth/arcHeight.
   * @public
   *
   * @param {x} number
   * @param {y} number
   * @param {width} number
   * @param {height} number
   * @param {arcWidth} number
   * @param {arcHeight} number
   * @param {Object} [options]
   * @returns {Rectangle}
   */
  Rectangle.roundedRect = function( x, y, width, height, arcWidth, arcHeight, options ) {
    return new Rectangle( x, y, width, height, arcWidth, arcHeight, options );
  };

  /**
   * Creates a rectangle x/y/width/height matching the specified bounds.
   * @public
   *
   * @param {Bounds2} bounds
   * @param {Object} [options]
   * @returns {Rectangle}
   */
  Rectangle.bounds = function( bounds, options ) {
    return new Rectangle( bounds.minX, bounds.minY, bounds.width, bounds.height, 0, 0, options );
  };

  /**
   * Creates a rounded rectangle x/y/width/height matching the specified bounds (Rectangle.bounds, but with additional
   * arcWidth and arcHeight).
   * @public
   *
   * @param {Bounds2} bounds
   * @param {number} arcWidth
   * @param {number} arcHeight
   * @param {Object} [options]
   * @returns {Rectangle}
   */
  Rectangle.roundedBounds = function( bounds, arcWidth, arcHeight, options ) {
    return new Rectangle( bounds.minX, bounds.minY, bounds.width, bounds.height, arcWidth, arcHeight, options );
  };

  /**
   * Creates a rectangle with top/left of (0,0) with the specified {Dimension2}'s width and height.
   * @public
   *
   * @param {Dimension2} dimension
   * @param {Object} [options]
   * @returns {Rectangle}
   */
  Rectangle.dimension = function( dimension, options ) {
    return new Rectangle( 0, 0, dimension.width, dimension.height, 0, 0, options );
  };

  /*---------------------------------------------------------------------------*
   * Rendering state mixin (DOM/SVG)
   *----------------------------------------------------------------------------*/

  /**
   * A mixin to drawables for Rectangle that need to store state about what the current display is currently showing,
   * so that updates to the Rectangle will only be made on attributes that specifically changed (and no change will be
   * necessary for an attribute that changed back to its original/currently-displayed value). Generally, this is used
   * for DOM and SVG drawables.
   *
   * This mixin assumes the PaintableStateful mixin is also mixed (always the case for Rectangle stateful drawables).
   */
  Rectangle.RectangleStatefulDrawable = {
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

      /**
       * Initializes the stateful mixin state, starting its "lifetime" until it is disposed with disposeState().
       * @protected
       *
       * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
       * @param {Instance} instance
       * @returns {RectangleStatefulDrawable} - Returns 'this' reference, for chaining
       */
      proto.initializeState = function( renderer, instance ) {
        // @protected {boolean} - Flag marked as true if ANY of the drawable dirty flags are set (basically everything except for transforms, as we
        //                        need to accelerate the transform case.
        this.paintDirty = true;
        this.dirtyX = true;
        this.dirtyY = true;
        this.dirtyWidth = true;
        this.dirtyHeight = true;
        this.dirtyCornerXRadius = true;
        this.dirtyCornerYRadius = true;

        // After adding flags, we'll initialize the mixed-in PaintableStateful state.
        this.initializePaintableState( renderer, instance );

        return this; // allow for chaining
      };

      /**
       * Disposes the stateful mixin state, so it can be put into the pool to be initialized again.
       * @protected
       */
      proto.disposeState = function() {
        this.disposePaintableState();
      };

      /**
       * A "catch-all" dirty method that directly marks the paintDirty flag and triggers propagation of dirty
       * information. This can be used by other mark* methods, or directly itself if the paintDirty flag is checked.
       * @public (scenery-internal)
       *
       * It should be fired (indirectly or directly) for anything besides transforms that needs to make a drawable
       * dirty.
       */
      proto.markPaintDirty = function() {
        this.paintDirty = true;
        this.markDirty();
      };

      proto.markDirtyRectangle = function() {
        // TODO: consider bitmask instead?
        this.dirtyX = true;
        this.dirtyY = true;
        this.dirtyWidth = true;
        this.dirtyHeight = true;
        this.dirtyCornerXRadius = true;
        this.dirtyCornerYRadius = true;
        this.markPaintDirty();
      };

      proto.markDirtyX = function() {
        this.dirtyX = true;
        this.markPaintDirty();
      };
      proto.markDirtyY = function() {
        this.dirtyY = true;
        this.markPaintDirty();
      };
      proto.markDirtyWidth = function() {
        this.dirtyWidth = true;
        this.markPaintDirty();
      };
      proto.markDirtyHeight = function() {
        this.dirtyHeight = true;
        this.markPaintDirty();
      };
      proto.markDirtyCornerXRadius = function() {
        this.dirtyCornerXRadius = true;
        this.markPaintDirty();
      };
      proto.markDirtyCornerYRadius = function() {
        this.dirtyCornerYRadius = true;
        this.markPaintDirty();
      };

      /**
       * Clears all of the dirty flags (after they have been checked), so that future mark* methods will be able to flag them again.
       * @public (scenery-internal)
       */
      proto.setToCleanState = function() {
        this.paintDirty = false;
        this.dirtyX = false;
        this.dirtyY = false;
        this.dirtyWidth = false;
        this.dirtyHeight = false;
        this.dirtyCornerXRadius = false;
        this.dirtyCornerYRadius = false;
      };

      Paintable.PaintableStatefulDrawable.mixin( drawableType );
    }
  };

  /*---------------------------------------------------------------------------*
   * DOM rendering
   *----------------------------------------------------------------------------*/

  /**
   * A generated DOMSelfDrawable whose purpose will be drawing our Rectangle. One of these drawables will be created
   * for each displayed instance of a Rectangle.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Rectangle.RectangleDOMDrawable = function RectangleDOMDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( DOMSelfDrawable, Rectangle.RectangleDOMDrawable, {
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
     * @returns {RectangleDOMDrawable} - Returns 'this' reference, for chaining
     */
    initialize: function( renderer, instance ) {
      // Super-type initialization
      this.initializeDOMSelfDrawable( renderer, instance );

      // Stateful mix-in initialization
      this.initializeState( renderer, instance );

      // only create elements if we don't already have them (we pool visual states always, and depending on the platform may also pool the actual elements to minimize
      // allocation and performance costs)
      if ( !this.fillElement || !this.strokeElement ) {
        var fillElement = this.fillElement = document.createElement( 'div' );
        fillElement.style.display = 'block';
        fillElement.style.position = 'absolute';
        fillElement.style.left = '0';
        fillElement.style.top = '0';
        fillElement.style.pointerEvents = 'none';

        var strokeElement = this.strokeElement = document.createElement( 'div' );
        strokeElement.style.display = 'block';
        strokeElement.style.position = 'absolute';
        strokeElement.style.left = '0';
        strokeElement.style.top = '0';
        strokeElement.style.pointerEvents = 'none';
        fillElement.appendChild( strokeElement );
      }

      this.domElement = this.fillElement;

      // Apply CSS needed for future CSS transforms to work properly.
      scenery.Util.prepareForTransform( this.domElement, this.forceAcceleration );

      return this; // allow for chaining
    },

    /**
     * Updates our DOM element so that its appearance matches our node's representation.
     * @protected
     *
     * This implements part of the DOMSelfDrawable required API for subtypes.
     */
    updateDOM: function() {
      var node = this.node;
      var fillElement = this.fillElement;
      var strokeElement = this.strokeElement;

      if ( this.paintDirty ) {
        var borderRadius = Math.min( node._cornerXRadius, node._cornerYRadius );
        var borderRadiusDirty = this.dirtyCornerXRadius || this.dirtyCornerYRadius;

        if ( this.dirtyWidth ) {
          fillElement.style.width = node._rectWidth + 'px';
        }
        if ( this.dirtyHeight ) {
          fillElement.style.height = node._rectHeight + 'px';
        }
        if ( borderRadiusDirty ) {
          fillElement.style[ Features.borderRadius ] = borderRadius + 'px'; // if one is zero, we are not rounded, so we do the min here
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
          // the other option would be to update stroked information when there is no stroke (major performance loss for fill-only rectangles)
          var hadNoStrokeBefore = this.lastStroke === null;

          if ( hadNoStrokeBefore || this.dirtyWidth || this.dirtyLineWidth ) {
            strokeElement.style.width = ( node._rectWidth - node.getLineWidth() ) + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyHeight || this.dirtyLineWidth ) {
            strokeElement.style.height = ( node._rectHeight - node.getLineWidth() ) + 'px';
          }
          if ( hadNoStrokeBefore || this.dirtyLineWidth ) {
            strokeElement.style.left = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.top = ( -node.getLineWidth() / 2 ) + 'px';
            strokeElement.style.borderWidth = node.getLineWidth() + 'px';
          }

          if ( hadNoStrokeBefore || this.dirtyStroke ) {
            strokeElement.style.borderColor = node.getSimpleCSSStroke();
          }

          if ( hadNoStrokeBefore || borderRadiusDirty || this.dirtyLineWidth || this.dirtyLineOptions ) {
            strokeElement.style[ Features.borderRadius ] = ( node.isRounded() || node.getLineJoin() === 'round' ) ? ( borderRadius + node.getLineWidth() / 2 ) + 'px' : '0';
          }
        }
      }

      // shift the element vertically, postmultiplied with the entire transform.
      if ( this.transformDirty || this.dirtyX || this.dirtyY ) {
        scratchMatrix.set( this.getTransformMatrix() );
        var translation = Matrix3.translation( node._rectX, node._rectY );
        scratchMatrix.multiplyMatrix( translation );
        translation.freeToPool();
        scenery.Util.applyPreparedTransform( scratchMatrix, this.fillElement, this.forceAcceleration );
      }

      // clear all of the dirty flags
      this.setToCleanState();
      this.cleanPaintableState();
      this.transformDirty = false;
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      this.disposeState();

      if ( !keepDOMRectangleElements ) {
        // clear the references
        this.fillElement = null;
        this.strokeElement = null;
        this.domElement = null;
      }

      DOMSelfDrawable.prototype.dispose.call( this );
    }
  } );
  Rectangle.RectangleStatefulDrawable.mixin( Rectangle.RectangleDOMDrawable );
  // This sets up RectangleDOMDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Rectangle.RectangleDOMDrawable );

  /*---------------------------------------------------------------------------*
   * SVG rendering
   *----------------------------------------------------------------------------*/

  /**
   * A generated SVGSelfDrawable whose purpose will be drawing our Rectangle. One of these drawables will be created
   * for each displayed instance of a Rectangle.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Rectangle.RectangleSVGDrawable = function RectangleSVGDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( SVGSelfDrawable, Rectangle.RectangleSVGDrawable, {
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
     * @returns {RectangleSVGDrawable} - Returns 'this' reference, for chaining
     */
    initialize: function( renderer, instance ) {
      // Super-type initialization
      this.initializeSVGSelfDrawable( renderer, instance, true, keepSVGRectangleElements ); // usesPaint: true

      this.lastArcW = -1; // invalid on purpose
      this.lastArcH = -1; // invalid on purpose

      // @protected {SVGRectElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
      this.svgElement = this.svgElement || document.createElementNS( scenery.svgns, 'rect' );

      return this;
    },

    updateSVGSelf: function() {
      var rect = this.svgElement;

      if ( this.dirtyX ) {
        rect.setAttribute( 'x', this.node._rectX );
      }
      if ( this.dirtyY ) {
        rect.setAttribute( 'y', this.node._rectY );
      }
      if ( this.dirtyWidth ) {
        rect.setAttribute( 'width', this.node._rectWidth );
      }
      if ( this.dirtyHeight ) {
        rect.setAttribute( 'height', this.node._rectHeight );
      }
      if ( this.dirtyCornerXRadius || this.dirtyCornerYRadius || this.dirtyWidth || this.dirtyHeight ) {
        var arcw = 0;
        var arch = 0;

        // workaround for various browsers if rx=20, ry=0 (behavior is inconsistent, either identical to rx=20,ry=20, rx=0,ry=0. We'll treat it as rx=0,ry=0)
        // see https://github.com/phetsims/scenery/issues/183
        if ( this.node.isRounded() ) {
          var maximumArcSize = this.node.getMaximumArcSize();
          arcw = Math.min( this.node._cornerXRadius, maximumArcSize );
          arch = Math.min( this.node._cornerYRadius, maximumArcSize );
        }
        if ( arcw !== this.lastArcW ) {
          this.lastArcW = arcw;
          rect.setAttribute( 'rx', arcw );
        }
        if ( arch !== this.lastArcH ) {
          this.lastArcH = arch;
          rect.setAttribute( 'ry', arch );
        }
      }

      this.updateFillStrokeStyle( rect );
    }
  } );
  Rectangle.RectangleStatefulDrawable.mixin( Rectangle.RectangleSVGDrawable );
  // This sets up RectangleSVGDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Rectangle.RectangleSVGDrawable );

  /*---------------------------------------------------------------------------*
   * Canvas rendering
   *----------------------------------------------------------------------------*/

  /**
   * A generated CanvasSelfDrawable whose purpose will be drawing our Rectangle. One of these drawables will be created
   * for each displayed instance of a Rectangle.
   * @constructor
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Rectangle.RectangleCanvasDrawable = function RectangleCanvasDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( CanvasSelfDrawable, Rectangle.RectangleCanvasDrawable, {
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
     * @returns {RectangleCanvasDrawable} - Returns 'this' reference, for chaining
     */
    initialize: function( renderer, instance ) {
      this.initializeCanvasSelfDrawable( renderer, instance );
      this.initializePaintableStateless( renderer, instance );
      return this;
    },

    /**
     * Convenience function for drawing a rectangular path (with our Rectangle node's parameters) to the Canvas context.
     * @private
     *
     * @param {CanvasRenderingContext2D} context - To execute drawing commands on.
     * @param {Node} node - The node whose rectangle we want to draw
     */
    writeRectangularPath: function( context, node ) {
      context.beginPath();
      context.moveTo( node._rectX, node._rectY );
      context.lineTo( node._rectX + node._rectWidth, node._rectY );
      context.lineTo( node._rectX + node._rectWidth, node._rectY + node._rectHeight );
      context.lineTo( node._rectX, node._rectY + node._rectHeight );
      context.closePath();
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

      // use the standard version if it's a rounded rectangle, since there is no Canvas-optimized version for that
      if ( node.isRounded() ) {
        context.beginPath();
        var maximumArcSize = node.getMaximumArcSize();
        var arcw = Math.min( node._cornerXRadius, maximumArcSize );
        var arch = Math.min( node._cornerYRadius, maximumArcSize );
        var lowX = node._rectX + arcw;
        var highX = node._rectX + node._rectWidth - arcw;
        var lowY = node._rectY + arch;
        var highY = node._rectY + node._rectHeight - arch;
        if ( arcw === arch ) {
          // we can use circular arcs, which have well defined stroked offsets
          context.arc( highX, lowY, arcw, -Math.PI / 2, 0, false );
          context.arc( highX, highY, arcw, 0, Math.PI / 2, false );
          context.arc( lowX, highY, arcw, Math.PI / 2, Math.PI, false );
          context.arc( lowX, lowY, arcw, Math.PI, Math.PI * 3 / 2, false );
        }
        else {
          // we have to resort to elliptical arcs
          context.ellipse( highX, lowY, arcw, arch, 0, -Math.PI / 2, 0, false );
          context.ellipse( highX, highY, arcw, arch, 0, 0, Math.PI / 2, false );
          context.ellipse( lowX, highY, arcw, arch, 0, Math.PI / 2, Math.PI, false );
          context.ellipse( lowX, lowY, arcw, arch, 0, Math.PI, Math.PI * 3 / 2, false );
        }
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
      }
      else {
        // TODO: how to handle fill/stroke delay optimizations here?
        if ( node.hasFill() ) {
          // If we need the fill pattern/gradient to have a different transformation, we can't use fillRect.
          // See https://github.com/phetsims/scenery/issues/543
          if ( node.getFillValue().transformMatrix ) {
            this.writeRectangularPath( context, node );
            node.beforeCanvasFill( wrapper ); // defined in Paintable
            context.fill();
            node.afterCanvasFill( wrapper ); // defined in Paintable
          }
          else {
            node.beforeCanvasFill( wrapper ); // defined in Paintable
            context.fillRect( node._rectX, node._rectY, node._rectWidth, node._rectHeight );
            node.afterCanvasFill( wrapper ); // defined in Paintable
          }
        }
        if ( node.hasStroke() ) {
          // If we need the fill pattern/gradient to have a different transformation, we can't use fillRect.
          // See https://github.com/phetsims/scenery/issues/543
          if ( node.getStrokeValue().transformMatrix ) {
            this.writeRectangularPath( context, node );
            node.beforeCanvasStroke( wrapper ); // defined in Paintable
            context.stroke();
            node.afterCanvasStroke( wrapper ); // defined in Paintable
          }
          else {
            node.beforeCanvasStroke( wrapper ); // defined in Paintable
            context.strokeRect( node._rectX, node._rectY, node._rectWidth, node._rectHeight );
            node.afterCanvasStroke( wrapper ); // defined in Paintable
          }
        }
      }
    },

    // stateless dirty functions
    markDirtyRectangle: function() { this.markPaintDirty(); },

    markDirtyX: function() {
      this.markDirtyRectangle();
    },
    markDirtyY: function() {
      this.markDirtyRectangle();
    },
    markDirtyWidth: function() {
      this.markDirtyRectangle();
    },
    markDirtyHeight: function() {
      this.markDirtyRectangle();
    },
    markDirtyCornerXRadius: function() {
      this.markDirtyRectangle();
    },
    markDirtyCornerYRadius: function() {
      this.markDirtyRectangle();
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      CanvasSelfDrawable.prototype.dispose.call( this );
      this.disposePaintableStateless();
    }
  } );
  Paintable.PaintableStatelessDrawable.mixin( Rectangle.RectangleCanvasDrawable );
  // This sets up RectangleCanvasDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Rectangle.RectangleCanvasDrawable );

  /*---------------------------------------------------------------------------*
   * WebGL rendering
   *----------------------------------------------------------------------------*/

  var scratchColor = new Color( 'transparent' );

  /**
   * A generated WebGLSelfDrawable whose purpose will be drawing our Rectangle. One of these drawables will be created
   * for each displayed instance of a Rectangle.
   * @constructor
   *
   * NOTE: This drawable currently only supports solid fills and no strokes.
   *
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  Rectangle.RectangleWebGLDrawable = function RectangleWebGLDrawable( renderer, instance ) {
    this.initialize( renderer, instance );
  };
  inherit( WebGLSelfDrawable, Rectangle.RectangleWebGLDrawable, {
    webglRenderer: Renderer.webglVertexColorPolygons,

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
     * @returns {RectangleWebGLDrawable} - Returns 'this' reference, for chaining
     */
    initialize: function( renderer, instance ) {
      this.initializeWebGLSelfDrawable( renderer, instance );

      // Stateful mix-in initialization
      this.initializeState( renderer, instance );

      if ( !this.vertexArray ) {
        // format [X Y R G B A] for all vertices
        this.vertexArray = new Float32Array( 6 * 6 ); // 6-length components for 6 vertices (2 tris).
      }

      // corner vertices in the relative transform root coordinate space
      this.upperLeft = new Vector2();
      this.lowerLeft = new Vector2();
      this.upperRight = new Vector2();
      this.lowerRight = new Vector2();

      this.transformDirty = true;
      this.includeVertices = true; // used by the processor

      return this;
    },

    onAddToBlock: function( webglBlock ) {
      this.webglBlock = webglBlock; // TODO: do we need this reference?
      this.markDirty();
    },

    onRemoveFromBlock: function( webglBlock ) {
    },

    // @override
    markTransformDirty: function() {
      this.transformDirty = true;

      WebGLSelfDrawable.prototype.markTransformDirty.call( this );
    },

    update: function() {
      if ( this.dirty ) {
        this.dirty = false;

        if ( this.dirtyFill ) {
          this.includeVertices = this.node.hasFill();

          if ( this.includeVertices ) {
            var fill = ( this.node.fill instanceof Property ) ? this.node.fill.value : this.node.fill;
            var color =  scratchColor.set( fill );
            var red = color.red / 255;
            var green = color.green / 255;
            var blue = color.blue / 255;
            var alpha = color.alpha;

            for ( var i = 0; i < 6; i++ ) {
              var offset = i * 6;
              this.vertexArray[ 2 + offset ] = red;
              this.vertexArray[ 3 + offset ] = green;
              this.vertexArray[ 4 + offset ] = blue;
              this.vertexArray[ 5 + offset ] = alpha;
            }
          }
        }

        if ( this.transformDirty || this.dirtyX || this.dirtyY || this.dirtyWidth || this.dirtyHeight ) {
          this.transformDirty = false;

          var x = this.node._rectX;
          var y = this.node._rectY;
          var width = this.node._rectWidth;
          var height = this.node._rectHeight;

          var transformMatrix = this.instance.relativeTransform.matrix; // with compute need, should always be accurate
          transformMatrix.multiplyVector2( this.upperLeft.setXY( x, y ) );
          transformMatrix.multiplyVector2( this.lowerLeft.setXY( x, y + height ) );
          transformMatrix.multiplyVector2( this.upperRight.setXY( x + width, y ) );
          transformMatrix.multiplyVector2( this.lowerRight.setXY( x + width, y + height ) );

          // first triangle XYs
          this.vertexArray[ 0 ] = this.upperLeft.x;
          this.vertexArray[ 1 ] = this.upperLeft.y;
          this.vertexArray[ 6 ] = this.lowerLeft.x;
          this.vertexArray[ 7 ] = this.lowerLeft.y;
          this.vertexArray[ 12 ] = this.upperRight.x;
          this.vertexArray[ 13 ] = this.upperRight.y;

          // second triangle XYs
          this.vertexArray[ 18 ] = this.upperRight.x;
          this.vertexArray[ 19 ] = this.upperRight.y;
          this.vertexArray[ 24 ] = this.lowerLeft.x;
          this.vertexArray[ 25 ] = this.lowerLeft.y;
          this.vertexArray[ 30 ] = this.lowerRight.x;
          this.vertexArray[ 31 ] = this.lowerRight.y;
        }
      }

      this.setToCleanState();
      this.cleanPaintableState();
    },

    /**
     * Disposes the drawable.
     * @public
     * @override
     */
    dispose: function() {
      // TODO: disposal of buffers?

      // super
      WebGLSelfDrawable.prototype.dispose.call( this );
    }
  } );
  Rectangle.RectangleStatefulDrawable.mixin( Rectangle.RectangleWebGLDrawable );
  // This sets up RectangleWebGLDrawable.createFromPool/dirtyFromPool and drawable.freeToPool() for the type, so
  // that we can avoid allocations by reusing previously-used drawables.
  SelfDrawable.Poolable.mixin( Rectangle.RectangleWebGLDrawable );

  return Rectangle;
} );
