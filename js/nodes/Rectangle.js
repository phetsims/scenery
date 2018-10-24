// Copyright 2013-2016, University of Colorado Boulder

/**
 * A rectangular node that inherits Path, and allows for optimized drawing and improved rectangle handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Bounds2 = require( 'DOT/Bounds2' );
  var Dimension2 = require( 'DOT/Dimension2' );
  var extendDefined = require( 'PHET_CORE/extendDefined' );
  var Features = require( 'SCENERY/util/Features' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Path = require( 'SCENERY/nodes/Path' );
  var RectangleCanvasDrawable = require( 'SCENERY/display/drawables/RectangleCanvasDrawable' );
  var RectangleDOMDrawable = require( 'SCENERY/display/drawables/RectangleDOMDrawable' );
  var RectangleSVGDrawable = require( 'SCENERY/display/drawables/RectangleSVGDrawable' );
  var RectangleWebGLDrawable = require( 'SCENERY/display/drawables/RectangleWebGLDrawable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var Shape = require( 'KITE/Shape' );

  var RECTANGLE_OPTION_KEYS = [
    'rectBounds', // Sets x/y/width/height based on bounds. See setRectBounds() for more documentation.
    'rectSize', // Sets width/height based on dimension. See setRectSize() for more documentation.
    'rectX', // Sets x. See setRectX() for more documentation.
    'rectY', // Sets y. See setRectY() for more documentation.
    'rectWidth', // Sets width. See setRectWidth() for more documentation.
    'rectHeight', // Sets height. See setRectHeight() for more documentation.
    'cornerRadius', // Sets corner radii. See setCornerRadius() for more documentation.
    'cornerXRadius', // Sets horizontal corner radius. See setCornerXRadius() for more documentation.
    'cornerYRadius' // Sets vertical corner radius. See setCornerYRadius() for more documentation.
  ];

  /**
   * @public
   * @constructor
   * @extends Path
   *
   * Possible constructor signatures
   * new Rectangle( x, y, width, height, cornerXRadius, cornerYRadius, [options] )
   * new Rectangle( x, y, width, height, [options] )
   * new Rectangle( [options] )
   * new Rectangle( bounds2, [options] )
   * new Rectangle( bounds2, cornerXRadius, cornerYRadius, [options] )
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
   *
   * Available parameters to the various constructor options:
   * @param {number} x - x-position of the upper-left corner (left bound)
   * @param {number} y - y-position of the upper-left corner (top bound)
   * @param {number} width - width of the rectangle to the right of the upper-left corner, required to be >= 0
   * @param {number} height - height of the rectangle below the upper-left corner, required to be >= 0
   * @param {number} cornerXRadius - positive vertical radius (width) of the rounded corner, or 0 to indicate the corner should be sharp
   * @param {number} cornerYRadius - positive horizontal radius (height) of the rounded corner, or 0 to indicate the corner should be sharp
   * @param {Object} [options] - Rectangle-specific options are documented in RECTANGLE_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
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
          assert && assert( y === undefined || Object.getPrototypeOf( y ) === Object.prototype,
            'Extra prototype on Node options object is a code smell' );

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
          assert && assert( height === undefined || Object.getPrototypeOf( height ) === Object.prototype,
            'Extra prototype on Node options object is a code smell' );

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
      assert && assert( cornerXRadius === undefined || Object.getPrototypeOf( cornerXRadius ) === Object.prototype,
        'Extra prototype on Node options object is a code smell' );

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
      assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
        'Extra prototype on Node options object is a code smell' );

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
    _mutatorKeys: RECTANGLE_OPTION_KEYS.concat( Path.prototype._mutatorKeys ),

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
     * @returns {Rectangle} - For chaining
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

      return this;
    },

    /**
     * Sets the x coordinate of the left side of this rectangle (in the local coordinate frame).
     * @public
     *
     * @param {number} x
     */
    setRectX: function( x ) {
      assert && assert( typeof x === 'number' && isFinite( x ), 'rectX should be a finite number' );

      if ( this._rectX !== x ) {
        this._rectX = x;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyX();
        }

        this.invalidateRectangle();
      }
      return this;
    },
    set rectX( value ) { this.setRectX( value ); },

    /**
     * Returns the x coordinate of the left side of this rectangle (in the local coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getRectX: function() {
      return this._rectX;
    },
    get rectX() { return this.getRectX(); },

    /**
     * Sets the y coordinate of the top side of this rectangle (in the local coordinate frame).
     * @public
     *
     * @param {number} y
     */
    setRectY: function( y ) {
      assert && assert( typeof y === 'number' && isFinite( y ), 'rectY should be a finite number' );

      if ( this._rectY !== y ) {
        this._rectY = y;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyY();
        }

        this.invalidateRectangle();
      }
      return this;
    },
    set rectY( value ) { this.setRectY( value ); },

    /**
     * Returns the y coordinate of the top side of this rectangle (in the local coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getRectY: function() {
      return this._rectY;
    },
    get rectY() { return this.getRectY(); },

    /**
     * Sets the width of the rectangle (in the local coordinate frame).
     * @public
     *
     * @param {number} width
     */
    setRectWidth: function( width ) {
      assert && assert( typeof width === 'number' && isFinite( width ), 'rectWidth should be a finite number' );

      if ( this._rectWidth !== width ) {
        this._rectWidth = width;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyWidth();
        }

        this.invalidateRectangle();
      }
      return this;
    },
    set rectWidth( value ) { this.setRectWidth( value ); },

    /**
     * Returns the width of the rectangle (in the local coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getRectWidth: function() {
      return this._rectWidth;
    },
    get rectWidth() { return this.getRectWidth(); },

    /**
     * Sets the height of the rectangle (in the local coordinate frame).
     * @public
     *
     * @param {number} height
     */
    setRectHeight: function( height ) {
      assert && assert( typeof height === 'number' && isFinite( height ), 'rectHeight should be a finite number' );

      if ( this._rectHeight !== height ) {
        this._rectHeight = height;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyHeight();
        }

        this.invalidateRectangle();
      }
      return this;
    },
    set rectHeight( value ) { this.setRectHeight( value ); },

    /**
     * Returns the height of the rectangle (in the local coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getRectHeight: function() {
      return this._rectHeight;
    },
    get rectHeight() { return this.getRectHeight(); },

    /**
     * Sets the horizontal corner radius of the rectangle (in the local coordinate frame).
     * @public
     *
     * If the cornerXRadius and cornerYRadius are the same, the corners will be rounded circular arcs with that radius
     * (or a smaller radius if the rectangle is too small).
     *
     * If the cornerXRadius and cornerYRadius are different, the corners will be elliptical arcs, and the horizontal
     * radius will be equal to cornerXRadius (or a smaller radius if the rectangle is too small).
     *
     * @param {number} radius
     */
    setCornerXRadius: function( radius ) {
      assert && assert( typeof radius === 'number' && isFinite( radius ), 'cornerXRadius should be a finite number' );

      if ( this._cornerXRadius !== radius ) {
        this._cornerXRadius = radius;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyCornerXRadius();
        }

        this.invalidateRectangle();
      }
      return this;
    },
    set cornerXRadius( value ) { this.setCornerXRadius( value ); },

    /**
     * Returns the horizontal corner radius of the rectangle (in the local coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getCornerXRadius: function() {
      return this._cornerXRadius;
    },
    get cornerXRadius() { return this.getCornerXRadius(); },

    /**
     * Sets the vertical corner radius of the rectangle (in the local coordinate frame).
     * @public
     *
     * If the cornerXRadius and cornerYRadius are the same, the corners will be rounded circular arcs with that radius
     * (or a smaller radius if the rectangle is too small).
     *
     * If the cornerXRadius and cornerYRadius are different, the corners will be elliptical arcs, and the vertical
     * radius will be equal to cornerYRadius (or a smaller radius if the rectangle is too small).
     *
     * @param {number} radius
     */
    setCornerYRadius: function( radius ) {
      assert && assert( typeof radius === 'number' && isFinite( radius ), 'cornerYRadius should be a finite number' );

      if ( this._cornerYRadius !== radius ) {
        this._cornerYRadius = radius;

        var stateLen = this._drawables.length;
        for ( var i = 0; i < stateLen; i++ ) {
          this._drawables[ i ].markDirtyCornerYRadius();
        }

        this.invalidateRectangle();
      }
      return this;
    },
    set cornerYRadius( value ) { this.setCornerYRadius( value ); },

    /**
     * Returns the vertical corner radius of the rectangle (in the local coordinate frame).
     * @public
     *
     * @returns {number}
     */
    getCornerYRadius: function() {
      return this._cornerYRadius;
    },
    get cornerYRadius() { return this.getCornerYRadius(); },

    /**
     * Sets the Rectangle's x/y/width/height from the Bounds2 passed in.
     * @public
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

    /**
     * Sets the width of the rectangle while keeping its right edge (x + width) in the same position
     * @public
     *
     * @param {number} width - New width for the rectangle
     * @returns {Rectangle} - For chaining
     */
    setRectWidthFromRight: function( width ) {
      assert && assert( typeof width === 'number' );

      if ( this._rectWidth !== width ) {
        var right = this._rectX + this._rectWidth;
        this.setRectWidth( width );
        this.setRectX( right - width );
      }

      return this;
    },
    set rectWidthFromRight( value ) { this.setRectWidthFromRight( value ); },
    get rectWidthFromRight() { return this.getRectWidth(); }, // because JSHint complains

    /**
     * Sets the height of the rectangle while keeping its bottom edge (y + height) in the same position
     * @public
     *
     * @param {number} height - New height for the rectangle
     * @returns {Rectangle} - For chaining
     */
    setRectHeightFromBottom: function( height ) {
      assert && assert( typeof height === 'number' );

      if ( this._rectHeight !== height ) {
        var bottom = this._rectY + this._rectHeight;
        this.setRectHeight( height );
        this.setRectY( bottom - height );
      }

      return this;
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
     * Returns a Shape that is equivalent to our rendered display. Generally used to lazily create a Shape instance
     * when one is needed, without having to do so beforehand.
     * @private
     *
     * @returns {Shape}
     */
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

    /**
     * Notifies that the rectangle has changed, and invalidates path information and our cached shape.
     * @protected
     */
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
     * @param {Matrix3} matrix - The transformation matrix already applied to the context.
     */
    canvasPaintSelf: function( wrapper, matrix ) {
      //TODO: Have a separate method for this, instead of touching the prototype. Can make 'this' references too easily.
      RectangleCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
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
     * It is impossible to set another shape on this Path subtype, as its effective shape is determined by other
     * parameters.
     * @public
     * @override
     *
     * @param {Shape|null} shape - Throws an error if it is not null.
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

    /**
     * Sets both of the corner radii to the same value, so that the rounded corners will be circular arcs.
     * @public
     *
     * @param {number} cornerRadius - The radius of the corners
     * @returns {Rectangle} - For chaining
     */
    setCornerRadius: function( cornerRadius ) {
      this.setCornerXRadius( cornerRadius );
      this.setCornerYRadius( cornerRadius );
      return this;
    },
    set cornerRadius( value ) { this.setCornerRadius( value ); },

    /**
     * Returns the corner radius if both the horizontal and vertical corner radii are the same.
     * @public
     *
     * NOTE: If there are different horizontal and vertical corner radii, this will fail an assertion and return the horizontal radius.
     *
     * @returns {number}
     */
    getCornerRadius: function() {
      assert && assert( this._cornerXRadius === this._cornerYRadius,
        'getCornerRadius() invalid if x/y radii are different' );

      return this._cornerXRadius;
    },
    get cornerRadius() { return this.getCornerRadius(); }
  } );

  /**
   * Returns whether a point is within a rounded rectangle.
   * @public
   *
   * @param {number} x - X value of the left side of the rectangle
   * @param {number} y - Y value of the top side of the rectangle
   * @param {number} width - Width of the rectangle
   * @param {number} height - Height of the rectangle
   * @param {number} arcWidth - Horizontal corner radius of the rectangle
   * @param {number} arcHeight - Vertical corner radius of the rectangle
   * @param {Vector2} point - The point that may or may not be in the rounded rectangle
   * @returns {boolean}
   */
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
   * See Rectangle's constructor for detailed parameter information.
   *
   * @param {number} x
   * @param {number} y
   * @param {number} width
   * @param {number} height
   * @param {Object} [options]
   * @returns {Rectangle}
   */
  Rectangle.rect = function( x, y, width, height, options ) {
    return new Rectangle( x, y, width, height, 0, 0, options );
  };

  /**
   * Creates a rounded rectangle with the specified x/y/width/height/cornerXRadius/cornerYRadius.
   * @public
   *
   * See Rectangle's constructor for detailed parameter information.
   *
   * @param {number} x
   * @param {number} y
   * @param {number} width
   * @param {number} height
   * @param {number} cornerXRadius
   * @param {number} cornerYRadius
   * @param {Object} [options]
   * @returns {Rectangle}
   */
  Rectangle.roundedRect = function( x, y, width, height, cornerXRadius, cornerYRadius, options ) {
    return new Rectangle( x, y, width, height, cornerXRadius, cornerYRadius, options );
  };

  /**
   * Creates a rectangle x/y/width/height matching the specified bounds.
   * @public
   *
   * See Rectangle's constructor for detailed parameter information.
   *
   * @param {Bounds2} bounds
   * @param {Object} [options]
   * @returns {Rectangle}
   */
  Rectangle.bounds = function( bounds, options ) {
    return new Rectangle( bounds.minX, bounds.minY, bounds.width, bounds.height, options );
  };

  /**
   * Creates a rounded rectangle x/y/width/height matching the specified bounds (Rectangle.bounds, but with additional
   * cornerXRadius and cornerYRadius).
   * @public
   *
   * See Rectangle's constructor for detailed parameter information.
   *
   * @param {Bounds2} bounds
   * @param {number} cornerXRadius
   * @param {number} cornerYRadius
   * @param {Object} [options]
   * @returns {Rectangle}
   */
  Rectangle.roundedBounds = function( bounds, cornerXRadius, cornerYRadius, options ) {
    return new Rectangle( bounds.minX, bounds.minY, bounds.width, bounds.height, cornerXRadius, cornerYRadius, options );
  };

  /**
   * Creates a rectangle with top/left of (0,0) with the specified {Dimension2}'s width and height.
   * @public
   *
   * See Rectangle's constructor for detailed parameter information.
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
