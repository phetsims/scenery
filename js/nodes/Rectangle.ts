// Copyright 2013-2021, University of Colorado Boulder

/**
 * A rectangular node that inherits Path, and allows for optimized drawing and improved rectangle handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Dimension2 from '../../../dot/js/Dimension2.js';
import Shape from '../../../kite/js/Shape.js';
import extendDefined from '../../../phet-core/js/extendDefined.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { scenery, Renderer, Features, Gradient, Pattern, Path, Instance, PathOptions, IRectangleDrawable, CanvasSelfDrawable, DOMSelfDrawable, SVGSelfDrawable, WebGLSelfDrawable, RectangleCanvasDrawable, RectangleDOMDrawable, RectangleSVGDrawable, RectangleWebGLDrawable, CanvasContextWrapper } from '../imports.js';
import Matrix3 from '../../../dot/js/Matrix3.js';

const RECTANGLE_OPTION_KEYS = [
  'rectBounds', // {Bounds2} - Sets x/y/width/height based on bounds. See setRectBounds() for more documentation.
  'rectSize', // {Dimension2} - Sets width/height based on dimension. See setRectSize() for more documentation.
  'rectX', // {number} - Sets x. See setRectX() for more documentation.
  'rectY', // {number} - Sets y. See setRectY() for more documentation.
  'rectWidth', // {number} - Sets width. See setRectWidth() for more documentation.
  'rectHeight', // Sets height. See setRectHeight() for more documentation.
  'cornerRadius', // {number} - Sets corner radii. See setCornerRadius() for more documentation.
  'cornerXRadius', // {number} - Sets horizontal corner radius. See setCornerXRadius() for more documentation.
  'cornerYRadius' // {number} - Sets vertical corner radius. See setCornerYRadius() for more documentation.
];

type RectangleOptions = {
  rectBounds?: Bounds2,
  rectSize?: Dimension2,
  rectX?: number,
  rectY?: number,
  rectWidth?: number,
  rectHeight?: number,
  cornerRadius?: number,
  cornerXRadius?: number,
  cornerYRadius?: number
} & PathOptions;

class Rectangle extends Path {
  // X value of the left side of the rectangle
  _rectX: number;

  // Y value of the top side of the rectangle
  _rectY: number;

  // Width of the rectangle
  _rectWidth: number;

  // Height of the rectangle
  _rectHeight: number;

  // X radius of rounded corners
  _cornerXRadius: number;

  // Y radius of rounded corners
  _cornerYRadius: number;

  /**
   * @public
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
   * @param x - x-position of the upper-left corner (left bound)
   * @param [y] - y-position of the upper-left corner (top bound)
   * @param [width] - width of the rectangle to the right of the upper-left corner, required to be >= 0
   * @param [height] - height of the rectangle below the upper-left corner, required to be >= 0
   * @param [cornerXRadius] - positive vertical radius (width) of the rounded corner, or 0 to indicate the corner should be sharp
   * @param [cornerYRadius] - positive horizontal radius (height) of the rounded corner, or 0 to indicate the corner should be sharp
   * @param [options] - Rectangle-specific options are documented in RECTANGLE_OPTION_KEYS above, and can be provided
   *                             along-side options for Node
   */
  constructor( options?: RectangleOptions );
  constructor( bounds: Bounds2, options?: RectangleOptions );
  constructor( bounds: Bounds2, cornerRadiusX: number, cornerRadiusY: number, options?: RectangleOptions );
  constructor( x: number, y: number, width: number, height: number, options?: RectangleOptions );
  constructor( x: number, y: number, width: number, height: number, cornerXRadius: number, cornerYRadius: number, options?: RectangleOptions );
  constructor( x?: number | Bounds2 | RectangleOptions, y?: number | RectangleOptions, width?: number, height?: number | RectangleOptions, cornerXRadius?: number | RectangleOptions, cornerYRadius?: number, options?: RectangleOptions ) {
    super( null );

    this._rectX = 0;
    this._rectY = 0;
    this._rectWidth = 0;
    this._rectHeight = 0;
    this._cornerXRadius = 0;
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

    this.mutate( options );
  }


  /**
   * Determines the maximum arc size that can be accommodated by the current width and height.
   *
   * If the corner radii are the same as the maximum arc size on a square, it will appear to be a circle (the arcs
   * take up all of the room, and leave no straight segments). In the case of a non-square, one direction of edges
   * will exist (e.g. top/bottom or left/right), while the other edges would be fully rounded.
   */
  private getMaximumArcSize(): number {
    return Math.min( this._rectWidth / 2, this._rectHeight / 2 );
  }

  /**
   * Determines the default allowed renderers (returned via the Renderer bitmask) that are allowed, given the
   * current stroke options. (scenery-internal)
   *
   * We can support the DOM renderer if there is a solid-styled stroke with non-bevel line joins
   * (which otherwise wouldn't be supported).
   *
   * @returns - Renderer bitmask, see Renderer for details
   */
  getStrokeRendererBitmask(): number {
    let bitmask = super.getStrokeRendererBitmask();
    const stroke = this.getStroke();
    // DOM stroke handling doesn't YET support gradients, patterns, or dashes (with the current implementation, it shouldn't be too hard)
    if ( stroke && !( stroke instanceof Gradient ) && !( stroke instanceof Pattern ) && !this.hasLineDash() ) {
      // we can't support the bevel line-join with our current DOM rectangle display
      if ( this.getLineJoin() === 'miter' || ( this.getLineJoin() === 'round' && Features.borderRadius ) ) {
        bitmask |= Renderer.bitmaskDOM;
      }
    }

    if ( !this.hasStroke() ) {
      bitmask |= Renderer.bitmaskWebGL;
    }

    return bitmask;
  }

  /**
   * Determines the allowed renderers that are allowed (or excluded) based on the current Path. (scenery-internal)
   *
   * @returns - Renderer bitmask, see Renderer for details
   */
  getPathRendererBitmask(): number {
    let bitmask = Renderer.bitmaskCanvas | Renderer.bitmaskSVG;

    const maximumArcSize = this.getMaximumArcSize();

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
  }

  /**
   * Sets all of the shape-determining parameters for the rectangle.
   *
   * @param x - The x-position of the left side of the rectangle.
   * @param y - The y-position of the top side of the rectangle.
   * @param width - The width of the rectangle.
   * @param height - The height of the rectangle.
   * @param [cornerXRadius] - The horizontal radius of curved corners (0 for sharp corners)
   * @param [cornerYRadius] - The vertical radius of curved corners (0 for sharp corners)
   */
  setRect( x: number, y: number, width: number, height: number, cornerXRadius?: number, cornerYRadius?: number ): this {
    const hasXRadius = cornerXRadius !== undefined;
    const hasYRadius = cornerYRadius !== undefined;

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
      return this;
    }

    this._rectX = x;
    this._rectY = y;
    this._rectWidth = width;
    this._rectHeight = height;
    this._cornerXRadius = hasXRadius ? cornerXRadius : this._cornerXRadius;
    this._cornerYRadius = hasYRadius ? cornerYRadius : this._cornerYRadius;

    const stateLen = this._drawables.length;
    for ( let i = 0; i < stateLen; i++ ) {
      ( this._drawables[ i ] as unknown as IRectangleDrawable ).markDirtyRectangle();
    }
    this.invalidateRectangle();

    return this;
  }

  /**
   * Sets the x coordinate of the left side of this rectangle (in the local coordinate frame).
   */
  setRectX( x: number ): this {
    assert && assert( typeof x === 'number' && isFinite( x ), 'rectX should be a finite number' );

    if ( this._rectX !== x ) {
      this._rectX = x;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as IRectangleDrawable ).markDirtyX();
      }

      this.invalidateRectangle();
    }
    return this;
  }

  set rectX( value: number ) { this.setRectX( value ); }

  /**
   * Returns the x coordinate of the left side of this rectangle (in the local coordinate frame).
   */
  getRectX(): number {
    return this._rectX;
  }

  get rectX(): number { return this.getRectX(); }

  /**
   * Sets the y coordinate of the top side of this rectangle (in the local coordinate frame).
   */
  setRectY( y: number ): this {
    assert && assert( typeof y === 'number' && isFinite( y ), 'rectY should be a finite number' );

    if ( this._rectY !== y ) {
      this._rectY = y;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as IRectangleDrawable ).markDirtyY();
      }

      this.invalidateRectangle();
    }
    return this;
  }

  set rectY( value: number ) { this.setRectY( value ); }

  /**
   * Returns the y coordinate of the top side of this rectangle (in the local coordinate frame).
   */
  getRectY(): number {
    return this._rectY;
  }

  get rectY(): number { return this.getRectY(); }

  /**
   * Sets the width of the rectangle (in the local coordinate frame).
   */
  setRectWidth( width: number ): this {
    assert && assert( typeof width === 'number' && isFinite( width ), 'rectWidth should be a finite number' );

    if ( this._rectWidth !== width ) {
      this._rectWidth = width;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as IRectangleDrawable ).markDirtyWidth();
      }

      this.invalidateRectangle();
    }
    return this;
  }

  set rectWidth( value: number ) { this.setRectWidth( value ); }

  /**
   * Returns the width of the rectangle (in the local coordinate frame).
   */
  getRectWidth(): number {
    return this._rectWidth;
  }

  get rectWidth(): number { return this.getRectWidth(); }

  /**
   * Sets the height of the rectangle (in the local coordinate frame).
   */
  setRectHeight( height: number ): this {
    assert && assert( typeof height === 'number' && isFinite( height ), 'rectHeight should be a finite number' );

    if ( this._rectHeight !== height ) {
      this._rectHeight = height;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as IRectangleDrawable ).markDirtyHeight();
      }

      this.invalidateRectangle();
    }
    return this;
  }

  set rectHeight( value: number ) { this.setRectHeight( value ); }

  /**
   * Returns the height of the rectangle (in the local coordinate frame).
   */
  getRectHeight(): number {
    return this._rectHeight;
  }

  get rectHeight(): number { return this.getRectHeight(); }

  /**
   * Sets the horizontal corner radius of the rectangle (in the local coordinate frame).
   *
   * If the cornerXRadius and cornerYRadius are the same, the corners will be rounded circular arcs with that radius
   * (or a smaller radius if the rectangle is too small).
   *
   * If the cornerXRadius and cornerYRadius are different, the corners will be elliptical arcs, and the horizontal
   * radius will be equal to cornerXRadius (or a smaller radius if the rectangle is too small).
   */
  setCornerXRadius( radius: number ): this {
    assert && assert( typeof radius === 'number' && isFinite( radius ), 'cornerXRadius should be a finite number' );

    if ( this._cornerXRadius !== radius ) {
      this._cornerXRadius = radius;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as IRectangleDrawable ).markDirtyCornerXRadius();
      }

      this.invalidateRectangle();
    }
    return this;
  }

  set cornerXRadius( value: number ) { this.setCornerXRadius( value ); }

  /**
   * Returns the horizontal corner radius of the rectangle (in the local coordinate frame).
   */
  getCornerXRadius(): number {
    return this._cornerXRadius;
  }

  get cornerXRadius(): number { return this.getCornerXRadius(); }

  /**
   * Sets the vertical corner radius of the rectangle (in the local coordinate frame).
   *
   * If the cornerXRadius and cornerYRadius are the same, the corners will be rounded circular arcs with that radius
   * (or a smaller radius if the rectangle is too small).
   *
   * If the cornerXRadius and cornerYRadius are different, the corners will be elliptical arcs, and the vertical
   * radius will be equal to cornerYRadius (or a smaller radius if the rectangle is too small).
   */
  setCornerYRadius( radius: number ): this {
    assert && assert( typeof radius === 'number' && isFinite( radius ), 'cornerYRadius should be a finite number' );

    if ( this._cornerYRadius !== radius ) {
      this._cornerYRadius = radius;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as IRectangleDrawable ).markDirtyCornerYRadius();
      }

      this.invalidateRectangle();
    }
    return this;
  }

  set cornerYRadius( value: number ) { this.setCornerYRadius( value ); }

  /**
   * Returns the vertical corner radius of the rectangle (in the local coordinate frame).
   */
  getCornerYRadius(): number {
    return this._cornerYRadius;
  }

  get cornerYRadius(): number { return this.getCornerYRadius(); }

  /**
   * Sets the Rectangle's x/y/width/height from the Bounds2 passed in.
   */
  setRectBounds( bounds: Bounds2 ): this {
    assert && assert( bounds instanceof Bounds2 );

    this.setRect( bounds.x, bounds.y, bounds.width, bounds.height );

    return this;
  }

  set rectBounds( value: Bounds2 ) { this.setRectBounds( value ); }

  /**
   * Returns a new Bounds2 generated from this Rectangle's x/y/width/height.
   */
  getRectBounds(): Bounds2 {
    return Bounds2.rect( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
  }

  get rectBounds(): Bounds2 { return this.getRectBounds(); }

  /**
   * Sets the Rectangle's width/height from the Dimension2 size passed in.
   */
  setRectSize( size: Dimension2 ): this {
    assert && assert( size instanceof Dimension2 );

    this.setRectWidth( size.width );
    this.setRectHeight( size.height );

    return this;
  }

  set rectSize( value: Dimension2 ) { this.setRectSize( value ); }

  /**
   * Returns a new Dimension2 generated from this Rectangle's width/height.
   */
  getRectSize(): Dimension2 {
    return new Dimension2( this._rectWidth, this._rectHeight );
  }

  get rectSize(): Dimension2 { return this.getRectSize(); }

  /**
   * Sets the width of the rectangle while keeping its right edge (x + width) in the same position
   */
  setRectWidthFromRight( width: number ): this {
    assert && assert( typeof width === 'number' );

    if ( this._rectWidth !== width ) {
      const right = this._rectX + this._rectWidth;
      this.setRectWidth( width );
      this.setRectX( right - width );
    }

    return this;
  }

  set rectWidthFromRight( value: number ) { this.setRectWidthFromRight( value ); }

  get rectWidthFromRight(): number { return this.getRectWidth(); } // because JSHint complains

  /**
   * Sets the height of the rectangle while keeping its bottom edge (y + height) in the same position
   */
  setRectHeightFromBottom( height: number ): this {
    assert && assert( typeof height === 'number' );

    if ( this._rectHeight !== height ) {
      const bottom = this._rectY + this._rectHeight;
      this.setRectHeight( height );
      this.setRectY( bottom - height );
    }

    return this;
  }

  set rectHeightFromBottom( value: number ) { this.setRectHeightFromBottom( value ); }

  get rectHeightFromBottom(): number { return this.getRectHeight(); } // because JSHint complains

  /**
   * Returns whether this rectangle has any rounding applied at its corners. If either the x or y corner radius is 0,
   * then there is no rounding applied.
   */
  isRounded(): boolean {
    return this._cornerXRadius !== 0 && this._cornerYRadius !== 0;
  }

  /**
   * Computes the bounds of the Rectangle, including any applied stroke. Overridden for efficiency.
   */
  computeShapeBounds(): Bounds2 {
    let bounds = new Bounds2( this._rectX, this._rectY, this._rectX + this._rectWidth, this._rectY + this._rectHeight );
    if ( this._stroke ) {
      // since we are axis-aligned, any stroke will expand our bounds by a guaranteed set amount
      bounds = bounds.dilated( this.getLineWidth() / 2 );
    }
    return bounds;
  }

  /**
   * Returns a Shape that is equivalent to our rendered display. Generally used to lazily create a Shape instance
   * when one is needed, without having to do so beforehand.
   */
  private createRectangleShape(): Shape {
    if ( this.isRounded() ) {
      // copy border-radius CSS behavior in Chrome, where the arcs won't intersect, in cases where the arc segments at full size would intersect each other
      const maximumArcSize = Math.min( this._rectWidth / 2, this._rectHeight / 2 );
      return Shape.roundRectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight,
        Math.min( maximumArcSize, this._cornerXRadius ), Math.min( maximumArcSize, this._cornerYRadius ) ).makeImmutable();
    }
    else {
      return Shape.rectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight ).makeImmutable();
    }
  }

  /**
   * Notifies that the rectangle has changed, and invalidates path information and our cached shape.
   */
  protected invalidateRectangle() {
    assert && assert( isFinite( this._rectX ), `A rectangle needs to have a finite x (${this._rectX})` );
    assert && assert( isFinite( this._rectY ), `A rectangle needs to have a finite y (${this._rectY})` );
    assert && assert( this._rectWidth >= 0 && isFinite( this._rectWidth ),
      `A rectangle needs to have a non-negative finite width (${this._rectWidth})` );
    assert && assert( this._rectHeight >= 0 && isFinite( this._rectHeight ),
      `A rectangle needs to have a non-negative finite height (${this._rectHeight})` );
    assert && assert( this._cornerXRadius >= 0 && isFinite( this._cornerXRadius ),
      `A rectangle needs to have a non-negative finite arcWidth (${this._cornerXRadius})` );
    assert && assert( this._cornerYRadius >= 0 && isFinite( this._cornerYRadius ),
      `A rectangle needs to have a non-negative finite arcHeight (${this._cornerYRadius})` );

    // sets our 'cache' to null, so we don't always have to recompute our shape
    this._shape = null;

    // should invalidate the path and ensure a redraw
    this.invalidatePath();

    // since we changed the rectangle arc width/height, it could make DOM work or not
    this.invalidateSupportedRenderers();
  }

  /**
   * Computes whether the provided point is "inside" (contained) in this Rectangle's self content, or "outside".
   *
   * Handles axis-aligned optionally-rounded rectangles, although can only do optimized computation if it isn't
   * rounded. If it IS rounded, we check if a corner computation is needed (usually isn't), and only need to check
   * one corner for that test.
   *
   * @param point - Considered to be in the local coordinate frame
   */
  containsPointSelf( point: Vector2 ): boolean {
    const x = this._rectX;
    const y = this._rectY;
    const width = this._rectWidth;
    const height = this._rectHeight;
    const arcWidth = this._cornerXRadius;
    const arcHeight = this._cornerYRadius;
    const halfLine = this.getLineWidth() / 2;

    let result = true;
    if ( this._strokePickable ) {
      // test the outer boundary if we are stroke-pickable (if also fill-pickable, this is the only test we need)
      const rounded = this.isRounded();
      if ( !rounded && this.getLineJoin() === 'bevel' ) {
        // fall-back for bevel
        return super.containsPointSelf( point );
      }
      const miter = this.getLineJoin() === 'miter' && !rounded;
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
  }

  /**
   * Returns whether this Rectangle's selfBounds is intersected by the specified bounds.
   *
   * @param bounds - Bounds to test, assumed to be in the local coordinate frame.
   */
  intersectsBoundsSelf( bounds: Bounds2 ): boolean {
    return !this.computeShapeBounds().intersection( bounds ).isEmpty();
  }

  /**
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   *
   * @param wrapper
   * @param matrix - The transformation matrix already applied to the context.
   */
  canvasPaintSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ) {
    //TODO: Have a separate method for this, instead of touching the prototype. Can make 'this' references too easily.
    RectangleCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
  }

  /**
   * Creates a DOM drawable for this Rectangle. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  createDOMDrawable( renderer: number, instance: Instance ): DOMSelfDrawable {
    // @ts-ignore
    return RectangleDOMDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a SVG drawable for this Rectangle. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  createSVGDrawable( renderer: number, instance: Instance ): SVGSelfDrawable {
    // @ts-ignore
    return RectangleSVGDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a Canvas drawable for this Rectangle. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  createCanvasDrawable( renderer: number, instance: Instance ): CanvasSelfDrawable {
    // @ts-ignore
    return RectangleCanvasDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a WebGL drawable for this Rectangle. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  createWebGLDrawable( renderer: number, instance: Instance ): WebGLSelfDrawable {
    // @ts-ignore
    return RectangleWebGLDrawable.createFromPool( renderer, instance );
  }

  /*---------------------------------------------------------------------------*
   * Miscellaneous
   *----------------------------------------------------------------------------*/

  /**
   * It is impossible to set another shape on this Path subtype, as its effective shape is determined by other
   * parameters.
   *
   * @param shape - Throws an error if it is not null.
   */
  setShape( shape: Shape | string | null ): this {
    if ( shape !== null ) {
      throw new Error( 'Cannot set the shape of a Rectangle to something non-null' );
    }
    else {
      // probably called from the Path constructor
      this.invalidatePath();
    }
    return this;
  }

  /**
   * Returns an immutable copy of this Path subtype's representation.
   *
   * NOTE: This is created lazily, so don't call it if you don't have to!
   */
  getShape(): Shape {
    if ( !this._shape ) {
      this._shape = this.createRectangleShape();
    }
    return this._shape;
  }

  /**
   * Returns whether this Path has an associated Shape (instead of no shape, represented by null)
   */
  hasShape(): boolean {
    return true;
  }

  /**
   * Sets both of the corner radii to the same value, so that the rounded corners will be circular arcs.
   */
  setCornerRadius( cornerRadius: number ): this {
    this.setCornerXRadius( cornerRadius );
    this.setCornerYRadius( cornerRadius );
    return this;
  }

  set cornerRadius( value: number ) { this.setCornerRadius( value ); }

  /**
   * Returns the corner radius if both the horizontal and vertical corner radii are the same.
   *
   * NOTE: If there are different horizontal and vertical corner radii, this will fail an assertion and return the horizontal radius.
   */
  getCornerRadius(): number {
    assert && assert( this._cornerXRadius === this._cornerYRadius,
      'getCornerRadius() invalid if x/y radii are different' );

    return this._cornerXRadius;
  }

  get cornerRadius(): number { return this.getCornerRadius(); }

  /**
   * Returns whether a point is within a rounded rectangle.
   *
   * @param x - X value of the left side of the rectangle
   * @param y - Y value of the top side of the rectangle
   * @param width - Width of the rectangle
   * @param height - Height of the rectangle
   * @param arcWidth - Horizontal corner radius of the rectangle
   * @param arcHeight - Vertical corner radius of the rectangle
   * @param point - The point that may or may not be in the rounded rectangle
   */
  static intersects( x: number, y: number, width: number, height: number, arcWidth: number, arcHeight: number, point: Vector2 ): boolean {
    const result = point.x >= x &&
                   point.x <= x + width &&
                   point.y >= y &&
                   point.y <= y + height;

    if ( !result || arcWidth <= 0 || arcHeight <= 0 ) {
      return result;
    }

    // copy border-radius CSS behavior in Chrome, where the arcs won't intersect, in cases where the arc segments at full size would intersect each other
    const maximumArcSize = Math.min( width / 2, height / 2 );
    arcWidth = Math.min( maximumArcSize, arcWidth );
    arcHeight = Math.min( maximumArcSize, arcHeight );

    // we are rounded and inside the logical rectangle (if it didn't have rounded corners)

    // closest corner arc's center (we assume the rounded rectangle's arcs are 90 degrees fully, and don't intersect)
    let closestCornerX;
    let closestCornerY;
    let guaranteedInside = false;

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
    let offsetX = point.x - closestCornerX;
    let offsetY = point.y - closestCornerY;

    // normalize the coordinates so now we are dealing with a unit circle
    // (technically arc, but we are guaranteed to be in the area covered by the arc, so we just consider the circle)
    // NOTE: we are rounded, so both arcWidth and arcHeight are non-zero (this is well defined)
    offsetX /= arcWidth;
    offsetY /= arcHeight;

    offsetX *= offsetX;
    offsetY *= offsetY;
    return offsetX + offsetY <= 1; // return whether we are in the rounded corner. see the formula for an ellipse
  }

  /**
   * Creates a rectangle with the specified x/y/width/height.
   *
   * See Rectangle's constructor for detailed parameter information.
   */
  static rect( x: number, y: number, width: number, height: number, options?: RectangleOptions ): Rectangle {
    return new Rectangle( x, y, width, height, 0, 0, options );
  }

  /**
   * Creates a rounded rectangle with the specified x/y/width/height/cornerXRadius/cornerYRadius.
   *
   * See Rectangle's constructor for detailed parameter information.
   */
  static roundedRect( x: number, y: number, width: number, height: number, cornerXRadius: number, cornerYRadius: number, options?: RectangleOptions ): Rectangle {
    return new Rectangle( x, y, width, height, cornerXRadius, cornerYRadius, options );
  }

  /**
   * Creates a rectangle x/y/width/height matching the specified bounds.
   *
   * See Rectangle's constructor for detailed parameter information.
   */
  static bounds( bounds: Bounds2, options?: RectangleOptions ): Rectangle {
    return new Rectangle( bounds.minX, bounds.minY, bounds.width, bounds.height, options );
  }

  /**
   * Creates a rounded rectangle x/y/width/height matching the specified bounds (Rectangle.bounds, but with additional
   * cornerXRadius and cornerYRadius).
   *
   * See Rectangle's constructor for detailed parameter information.
   */
  static roundedBounds( bounds: Bounds2, cornerXRadius: number, cornerYRadius: number, options?: RectangleOptions ): Rectangle {
    return new Rectangle( bounds.minX, bounds.minY, bounds.width, bounds.height, cornerXRadius, cornerYRadius, options );
  }

  /**
   * Creates a rectangle with top/left of (0,0) with the specified {Dimension2}'s width and height.
   *
   * See Rectangle's constructor for detailed parameter information.
   */
  static dimension( dimension: Dimension2, options?: RectangleOptions ): Rectangle {
    return new Rectangle( 0, 0, dimension.width, dimension.height, 0, 0, options );
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
Rectangle.prototype._mutatorKeys = [ ...RECTANGLE_OPTION_KEYS, ...Path.prototype._mutatorKeys ];

/**
 * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
 *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
 *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
 * @public (scenery-internal)
 * @override
 */
Rectangle.prototype.drawableMarkFlags = Path.prototype.drawableMarkFlags.concat( [ 'x', 'y', 'width', 'height', 'cornerXRadius', 'cornerYRadius' ] ).filter( flag => flag !== 'shape' );

scenery.register( 'Rectangle', Rectangle );

export { Rectangle as default };
export type { RectangleOptions };