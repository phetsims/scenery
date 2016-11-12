// Copyright 2013-2015, University of Colorado Boulder

/**
 * A rectangular node that inherits Path, and allows for optimized drawing and improved rectangle handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var extendDefined = require( 'PHET_CORE/extendDefined' );
  var scenery = require( 'SCENERY/scenery' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var Dimension2 = require( 'DOT/Dimension2' );
  var Features = require( 'SCENERY/util/Features' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var RectangleCanvasDrawable = require( 'SCENERY/display/drawables/RectangleCanvasDrawable' );
  var RectangleDOMDrawable = require( 'SCENERY/display/drawables/RectangleDOMDrawable' );
  var RectangleSVGDrawable = require( 'SCENERY/display/drawables/RectangleSVGDrawable' );
  var RectangleWebGLDrawable = require( 'SCENERY/display/drawables/RectangleWebGLDrawable' );

  /**
   * @constructor
   * @extends Path
   * @mixes Paintable
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
    // @private {number} - X value of the left side of the rectangle
    this._rectX = 0;

    // @private {number} - Y value of the top side of the rectangle
    this._rectY = 0;

    // @private {number} - Width of the rectangle
    this._rectWidth = 0;

    // @private {number} - Height of the rectangle
    this._rectHeight = 0;

    // @private {number} - X radius of rounded corners
    this._cornerXRadius = 0;

    // @private {number} - Y radius of rounded corners
    this._cornerYRadius = 0;

    if ( typeof x === 'object' ) {
      // allow new Rectangle( bounds2, { ... } ) or new Rectangle( bounds2, cornerXRadius, cornerYRadius, { ... } )
      if ( x instanceof Bounds2 ) {
        // new Rectangle( bounds2, { ... } )
        if ( typeof y !== 'number' ) {
          assert && assert( arguments.length === 1 || arguments.length === 2,
            'new Rectangle( bounds, { ... } ) should only take one or two arguments' );
          assert && assert( y === undefined || typeof y === 'object',
            'new Rectangle( bounds, { ... } ) second parameter should only ever be an options object' );

          options = extendDefined( {
            rectBounds: x
          }, y ); // Our options object would be at y
        }
        // Rectangle( bounds2, cornerXRadius, cornerYRadius, { ... } )
        else {
          assert && assert( arguments.length === 3 || arguments.length === 4,
            'new Rectangle( bounds, cornerXRadius, cornerYRadius, { ... } ) should only take three or four arguments' );
          assert && assert( height === undefined || typeof height === 'object',
            'new Rectangle( bounds, cornerXRadius, cornerYRadius, { ... } ) fourth parameter should only ever be an options object' );

          options = extendDefined( {
            rectBounds: x,
            cornerXRadius: y, // ignore Intellij warning, our cornerXRadius is the second parameter
            cornerYRadius: width // ignore Intellij warning, our cornerYRadius is the third parameter
          }, height ); // Our options object would be at height
        }
      }
      // allow new Rectangle( { rectX: x, rectY: y, rectWidth: width, rectHeight: height, ... } )
      else {
        options = x;
      }
    }
    // new Rectangle( x, y, width, height, { ... } )
    else if ( cornerYRadius === undefined ) {
      assert && assert( arguments.length === 4 || arguments.length === 5,
        'new Rectangle( x, y, width, height, { ... } ) should only take four or five arguments' );
      assert && assert( cornerXRadius === undefined || typeof cornerXRadius === 'object',
        'new Rectangle( x, y, width, height, { ... } ) fifth parameter should only ever be an options object' );

      options = extendDefined( {
        rectX: x,
        rectY: y,
        rectWidth: width,
        rectHeight: height
      }, cornerXRadius );
    }
    // new Rectangle( x, y, width, height, cornerXRadius, cornerYRadius, { ... } )
    else {
      assert && assert( arguments.length === 6 || arguments.length === 7,
        'new Rectangle( x, y, width, height, cornerXRadius, cornerYRadius{ ... } ) should only take six or seven arguments' );
      assert && assert( options === undefined || typeof options === 'object',
        'new Rectangle( x, y, width, height, cornerXRadius, cornerYRadius{ ... } ) seventh parameter should only ever be an options object' );

      options = extendDefined( {
        rectX: x,
        rectY: y,
        rectWidth: width,
        rectHeight: height,
        cornerXRadius: cornerXRadius,
        cornerYRadius: cornerYRadius
      }, options );
    }

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
    _mutatorKeys: [ 'rectBounds', 'rectSize', 'rectX', 'rectY', 'rectWidth', 'rectHeight',
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

    /**
     * Determines the maximum arc size that can be accomodated by the current width and height.
     * @private
     *
     * If the corner radii are the same as the maximum arc size on a square, it will appear to be a circle (the arcs
     * take up all of the room, and leave no straight segments). In the case of a non-square, one direction of edges
     * will exist (e.g. top/bottom or left/right), while the other edges would be fully rounded.
     *
     * @returns {number}
     */
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

    /**
     * Sets all of the shape-determining parameters for the rectangle.
     * @public
     *
     * @param {number} x - The x-position of the left side of the rectangle.
     * @param {number} y - The y-position of the top side of the rectangle.
     * @param {number} width - The width of the rectangle.
     * @param {number} height - The height of the rectangle.
     * @param {number} [cornerXRadius] - The horizontal radius of curved corners (0 for sharp corners)
     * @param {number} [cornerYRadius] - The vertical radius of curved corners (0 for sharp corners)
     */
    setRect: function( x, y, width, height, cornerXRadius, cornerYRadius ) {
      var hasXRadius = cornerXRadius !== undefined;
      var hasYRadius = cornerYRadius !== undefined;

      assert && assert( typeof x === 'number' && isFinite( x ) &&
                        typeof y === 'number' && isFinite( y ) &&
                        typeof width === 'number' && isFinite( width ) &&
                        typeof height === 'number' && isFinite( height ), 'x/y/width/height should be finite numbers' );
      assert && assert( !hasXRadius || ( typeof cornerXRadius === 'number' && isFinite( cornerXRadius ) ) &&
                        !hasYRadius || ( typeof cornerYRadius === 'number' && isFinite( cornerYRadius ) ),
                        'Corner radii (if provided) should be finite numbers' );

      // If this doesn't change the rectangle, don't notify about changes.
      if ( this._rectX === x &&
           this._rectY === y &&
           this._rectWidth === width &&
           this._rectHeight === height &&
           ( !hasXRadius || this._cornerXRadius === cornerXRadius ) &&
           ( !hasYRadius || this._cornerYRadius === cornerYRadius ) ) {
        return;
      }

      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._cornerXRadius = hasXRadius ? cornerXRadius : this._cornerXRadius;
      this._cornerYRadius = hasYRadius ? cornerYRadius : this._cornerYRadius;

      var stateLen = this._drawables.length;
      for ( var i = 0; i < stateLen; i++ ) {
        this._drawables[ i ].markDirtyRectangle();
      }
      this.invalidateRectangle();
    },

    /**
     * Sets the Rectangle's x/y/wdith/height from the Bounds2 passed in.
     * @public
     *
     * TODO: Note that it currently resets corner radii, see https://github.com/phetsims/scenery/issues/576
     *
     * @param {Bounds2} bounds
     * @returns {Rectangle} - For chaining
     */
    setRectBounds: function( bounds ) {
      assert && assert( bounds instanceof Bounds2 );

      this.setRect( bounds.x, bounds.y, bounds.width, bounds.height );

      return this;
    },
    set rectBounds( value ) { this.setRectBounds( value ); },

    /**
     * Returns a new Bounds2 generated from this Rectangle's x/y/width/height.
     * @public
     *
     * @returns {Bounds2}
     */
    getRectBounds: function() {
      return Bounds2.rect( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
    },
    get rectBounds() { return this.getRectBounds(); },

    /**
     * Sets the Rectangle's width/height from the Dimension2 size passed in.
     * @public
     *
     * @param {Dimension2} size
     * @returns {Rectangle} - For chaining
     */
    setRectSize: function( size ) {
      assert && assert( size instanceof Dimension2 );

      this.setRectWidth( size.width );
      this.setRectHeight( size.height );

      return this;
    },
    set rectSize( value ) { this.setRectSize( value ); },

    /**
     * Returns a new Dimension2 generated from this Rectangle's width/height.
     * @public
     *
     * @returns {Dimension2}
     */
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
      RectangleCanvasDrawable.prototype.paintCanvas( wrapper, this );
    },

    /**
     * Creates a DOM drawable for this Rectangle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {DOMSelfDrawable}
     */
    createDOMDrawable: function( renderer, instance ) {
      return RectangleDOMDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a SVG drawable for this Rectangle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {SVGSelfDrawable}
     */
    createSVGDrawable: function( renderer, instance ) {
      return RectangleSVGDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a Canvas drawable for this Rectangle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {CanvasSelfDrawable}
     */
    createCanvasDrawable: function( renderer, instance ) {
      return RectangleCanvasDrawable.createFromPool( renderer, instance );
    },

    /**
     * Creates a WebGL drawable for this Rectangle.
     * @public (scenery-internal)
     * @override
     *
     * @param {number} renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
     * @param {Instance} instance - Instance object that will be associated with the drawable
     * @returns {WebGLSelfDrawable}
     */
    createWebGLDrawable: function( renderer, instance ) {
      return RectangleWebGLDrawable.createFromPool( renderer, instance );
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

  return Rectangle;
} );
