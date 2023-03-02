// Copyright 2013-2023, University of Colorado Boulder

/**
 * Displays a (stroked) line. Inherits Path, and allows for optimized drawing and improved parameter handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import extendDefined from '../../../phet-core/js/extendDefined.js';
import { CanvasContextWrapper, CanvasSelfDrawable, TLineDrawable, Instance, LineCanvasDrawable, LineSVGDrawable, Path, PathOptions, Renderer, scenery, SVGSelfDrawable } from '../imports.js';

const LINE_OPTION_KEYS = [
  'p1', // {Vector2} - Start position
  'p2', // {Vector2} - End position
  'x1', // {number} - Start x position
  'y1', // {number} - Start y position
  'x2', // {number} - End x position
  'y2' // {number} - End y position
];

type SelfOptions = {
  p1?: Vector2;
  p2?: Vector2;
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
};
export type LineOptions = SelfOptions & StrictOmit<PathOptions, 'shape'>;

export default class Line extends Path {

  // The x coordinate of the start point (point 1)
  private _x1: number;

  // The Y coordinate of the start point (point 1)
  private _y1: number;

  // The x coordinate of the start point (point 2)
  private _x2: number;

  // The y coordinate of the start point (point 2)
  private _y2: number;

  public constructor( options?: LineOptions );
  public constructor( p1: Vector2, p2: Vector2, options?: LineOptions );
  public constructor( x1: number, y1: number, x2: number, y2: number, options?: LineOptions );
  public constructor( x1?: number | Vector2 | LineOptions, y1?: number | Vector2, x2?: number | LineOptions, y2?: number, options?: LineOptions ) {
    super( null );

    this._x1 = 0;
    this._y1 = 0;
    this._x2 = 0;
    this._y2 = 0;

    // Remap constructor parameters to options
    if ( typeof x1 === 'object' ) {
      if ( x1 instanceof Vector2 ) {
        // assumes Line( Vector2, Vector2, options ), where x2 is our options
        assert && assert( x2 === undefined || typeof x2 === 'object' );
        assert && assert( x2 === undefined || Object.getPrototypeOf( x2 ) === Object.prototype,
          'Extra prototype on Node options object is a code smell' );

        options = extendDefined( {
          // First Vector2 is under the x1 name
          x1: x1.x,
          y1: x1.y,
          // Second Vector2 is under the y1 name
          x2: ( y1 as Vector2 ).x,
          y2: ( y1 as Vector2 ).y,

          strokePickable: true
        }, x2 ); // Options object (if available) is under the x2 name
      }
      else {
        // assumes Line( { ... } ), init to zero for now
        assert && assert( y1 === undefined );

        // Options object is under the x1 name
        assert && assert( x1 === undefined || Object.getPrototypeOf( x1 ) === Object.prototype,
          'Extra prototype on Node options object is a code smell' );

        options = extendDefined( {
          strokePickable: true
        }, x1 ); // Options object (if available) is under the x1 name
      }
    }
    else {
      // new Line( x1, y1, x2, y2, [options] )
      assert && assert( x1 !== undefined &&
      typeof y1 === 'number' &&
      typeof x2 === 'number' &&
      typeof y2 === 'number' );
      assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
        'Extra prototype on Node options object is a code smell' );

      options = extendDefined( {
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        strokePickable: true
      }, options );
    }

    this.mutate( options );
  }

  /**
   * Set all of the line's x and y values.
   *
   * @param x1 - the start x coordinate
   * @param y1 - the start y coordinate
   * @param x2 - the end x coordinate
   * @param y2 - the end y coordinate
   */
  public setLine( x1: number, y1: number, x2: number, y2: number ): this {
    assert && assert( x1 !== undefined &&
    y1 !== undefined &&
    x2 !== undefined &&
    y2 !== undefined, 'parameters need to be defined' );

    this._x1 = x1;
    this._y1 = y1;
    this._x2 = x2;
    this._y2 = y2;

    const stateLen = this._drawables.length;
    for ( let i = 0; i < stateLen; i++ ) {
      const state = this._drawables[ i ];
      ( state as unknown as TLineDrawable ).markDirtyLine();
    }

    this.invalidateLine();

    return this;
  }

  /**
   * Set the line's first point's x and y values
   */
  public setPoint1( p1: Vector2 ): this;
  setPoint1( x1: number, y1: number ): this; // eslint-disable-line @typescript-eslint/explicit-member-accessibility
  setPoint1( x1: number | Vector2, y1?: number ): this {  // eslint-disable-line @typescript-eslint/explicit-member-accessibility
    if ( typeof x1 === 'number' ) {

      // setPoint1( x1, y1 );
      assert && assert( x1 !== undefined && y1 !== undefined, 'parameters need to be defined' );
      this._x1 = x1;
      this._y1 = y1!;
    }
    else {

      // setPoint1( Vector2 )
      assert && assert( x1.x !== undefined && x1.y !== undefined, 'parameters need to be defined' );
      this._x1 = x1.x;
      this._y1 = x1.y;
    }
    const numDrawables = this._drawables.length;
    for ( let i = 0; i < numDrawables; i++ ) {
      ( this._drawables[ i ] as unknown as TLineDrawable ).markDirtyP1();
    }
    this.invalidateLine();

    return this;
  }

  public set p1( point: Vector2 ) { this.setPoint1( point ); }

  public get p1(): Vector2 { return new Vector2( this._x1, this._y1 ); }

  /**
   * Set the line's second point's x and y values
   */
  public setPoint2( p1: Vector2 ): this;
  setPoint2( x2: number, y2: number ): this; // eslint-disable-line @typescript-eslint/explicit-member-accessibility
  setPoint2( x2: number | Vector2, y2?: number ): this {  // eslint-disable-line @typescript-eslint/explicit-member-accessibility
    if ( typeof x2 === 'number' ) {
      // setPoint2( x2, y2 );
      assert && assert( x2 !== undefined && y2 !== undefined, 'parameters need to be defined' );
      this._x2 = x2;
      this._y2 = y2!;
    }
    else {
      // setPoint2( Vector2 )
      assert && assert( x2.x !== undefined && x2.y !== undefined, 'parameters need to be defined' );
      this._x2 = x2.x;
      this._y2 = x2.y;
    }
    const numDrawables = this._drawables.length;
    for ( let i = 0; i < numDrawables; i++ ) {
      ( this._drawables[ i ] as unknown as TLineDrawable ).markDirtyP2();
    }
    this.invalidateLine();

    return this;
  }

  public set p2( point: Vector2 ) { this.setPoint2( point ); }

  public get p2(): Vector2 { return new Vector2( this._x2, this._y2 ); }

  /**
   * Sets the x coordinate of the first point of the line.
   */
  public setX1( x1: number ): this {
    if ( this._x1 !== x1 ) {
      this._x1 = x1;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as TLineDrawable ).markDirtyX1();
      }

      this.invalidateLine();
    }
    return this;
  }

  public set x1( value: number ) { this.setX1( value ); }

  public get x1(): number { return this.getX1(); }

  /**
   * Returns the x coordinate of the first point of the line.
   */
  public getX1(): number {
    return this._x1;
  }

  /**
   * Sets the y coordinate of the first point of the line.
   */
  public setY1( y1: number ): this {
    if ( this._y1 !== y1 ) {
      this._y1 = y1;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as TLineDrawable ).markDirtyY1();
      }

      this.invalidateLine();
    }
    return this;
  }

  public set y1( value: number ) { this.setY1( value ); }

  public get y1(): number { return this.getY1(); }

  /**
   * Returns the y coordinate of the first point of the line.
   */
  public getY1(): number {
    return this._y1;
  }

  /**
   * Sets the x coordinate of the second point of the line.
   */
  public setX2( x2: number ): this {
    if ( this._x2 !== x2 ) {
      this._x2 = x2;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as TLineDrawable ).markDirtyX2();
      }

      this.invalidateLine();
    }
    return this;
  }

  public set x2( value: number ) { this.setX2( value ); }

  public get x2(): number { return this.getX2(); }

  /**
   * Returns the x coordinate of the second point of the line.
   */
  public getX2(): number {
    return this._x2;
  }

  /**
   * Sets the y coordinate of the second point of the line.
   */
  public setY2( y2: number ): this {
    if ( this._y2 !== y2 ) {
      this._y2 = y2;

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as TLineDrawable ).markDirtyY2();
      }

      this.invalidateLine();
    }
    return this;
  }

  public set y2( value: number ) { this.setY2( value ); }

  public get y2(): number { return this.getY2(); }

  /**
   * Returns the y coordinate of the second point of the line.
   */
  public getY2(): number {
    return this._y2;
  }

  /**
   * Returns a Shape that is equivalent to our rendered display. Generally used to lazily create a Shape instance
   * when one is needed, without having to do so beforehand.
   */
  private createLineShape(): Shape {
    return Shape.lineSegment( this._x1, this._y1, this._x2, this._y2 ).makeImmutable();
  }

  /**
   * Notifies that the line has changed and invalidates path information and our cached shape.
   */
  private invalidateLine(): void {
    assert && assert( isFinite( this._x1 ), `A line needs to have a finite x1 (${this._x1})` );
    assert && assert( isFinite( this._y1 ), `A line needs to have a finite y1 (${this._y1})` );
    assert && assert( isFinite( this._x2 ), `A line needs to have a finite x2 (${this._x2})` );
    assert && assert( isFinite( this._y2 ), `A line needs to have a finite y2 (${this._y2})` );

    // sets our 'cache' to null, so we don't always have to recompute our shape
    this._shape = null;

    // should invalidate the path and ensure a redraw
    this.invalidatePath();
  }

  /**
   * Computes whether the provided point is "inside" (contained) in this Line's self content, or "outside".
   *
   * Since an unstroked Line contains no area, we can quickly shortcut this operation.
   *
   * @param point - Considered to be in the local coordinate frame
   */
  public override containsPointSelf( point: Vector2 ): boolean {
    if ( this._strokePickable ) {
      return super.containsPointSelf( point );
    }
    else {
      return false; // nothing is in a line! (although maybe we should handle edge points properly?)
    }
  }

  /**
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   *
   * @param wrapper
   * @param matrix - The transformation matrix already applied to the context.
   */
  protected override canvasPaintSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ): void {
    //TODO: Have a separate method for this, instead of touching the prototype. Can make 'this' references too easily.
    LineCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
  }

  /**
   * Computes the bounds of the Line, including any applied stroke. Overridden for efficiency.
   */
  public override computeShapeBounds(): Bounds2 {
    // optimized form for a single line segment (no joins, just two caps)
    if ( this._stroke ) {
      const lineCap = this.getLineCap();
      const halfLineWidth = this.getLineWidth() / 2;
      if ( lineCap === 'round' ) {
        // we can simply dilate by half the line width
        return new Bounds2(
          Math.min( this._x1, this._x2 ) - halfLineWidth, Math.min( this._y1, this._y2 ) - halfLineWidth,
          Math.max( this._x1, this._x2 ) + halfLineWidth, Math.max( this._y1, this._y2 ) + halfLineWidth );
      }
      else {
        // (dx,dy) is a vector p2-p1
        const dx = this._x2 - this._x1;
        const dy = this._y2 - this._y1;
        const magnitude = Math.sqrt( dx * dx + dy * dy );
        if ( magnitude === 0 ) {
          // if our line is a point, just dilate by halfLineWidth
          return new Bounds2( this._x1 - halfLineWidth, this._y1 - halfLineWidth, this._x2 + halfLineWidth, this._y2 + halfLineWidth );
        }
        // (sx,sy) is a vector with a magnitude of halfLineWidth pointed in the direction of (dx,dy)
        const sx = halfLineWidth * dx / magnitude;
        const sy = halfLineWidth * dy / magnitude;
        const bounds = Bounds2.NOTHING.copy();

        if ( lineCap === 'butt' ) {
          // four points just using the perpendicular stroked offsets (sy,-sx) and (-sy,sx)
          bounds.addCoordinates( this._x1 - sy, this._y1 + sx );
          bounds.addCoordinates( this._x1 + sy, this._y1 - sx );
          bounds.addCoordinates( this._x2 - sy, this._y2 + sx );
          bounds.addCoordinates( this._x2 + sy, this._y2 - sx );
        }
        else {
          assert && assert( lineCap === 'square' );

          // four points just using the perpendicular stroked offsets (sy,-sx) and (-sy,sx) and parallel stroked offsets
          bounds.addCoordinates( this._x1 - sx - sy, this._y1 - sy + sx );
          bounds.addCoordinates( this._x1 - sx + sy, this._y1 - sy - sx );
          bounds.addCoordinates( this._x2 + sx - sy, this._y2 + sy + sx );
          bounds.addCoordinates( this._x2 + sx + sy, this._y2 + sy - sx );
        }
        return bounds;
      }
    }
    else {
      // It might have a fill? Just include the fill bounds for now.
      const fillBounds = Bounds2.NOTHING.copy();
      fillBounds.addCoordinates( this._x1, this._y1 );
      fillBounds.addCoordinates( this._x2, this._y2 );
      return fillBounds;
    }
  }

  /**
   * Creates a SVG drawable for this Line.
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createSVGDrawable( renderer: number, instance: Instance ): SVGSelfDrawable {
    // @ts-expect-error
    return LineSVGDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a Canvas drawable for this Line.
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createCanvasDrawable( renderer: number, instance: Instance ): CanvasSelfDrawable {
    // @ts-expect-error
    return LineCanvasDrawable.createFromPool( renderer, instance );
  }

  /**
   * It is impossible to set another shape on this Path subtype, as its effective shape is determined by other
   * parameters.
   *
   * Throws an error if it is not null.
   */
  public override setShape( shape: Shape | null ): this {
    if ( shape !== null ) {
      throw new Error( 'Cannot set the shape of a Line to something non-null' );
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
  public override getShape(): Shape {
    if ( !this._shape ) {
      this._shape = this.createLineShape();
    }
    return this._shape;
  }

  /**
   * Returns whether this Path has an associated Shape (instead of no shape, represented by null)
   */
  public override hasShape(): boolean {
    return true;
  }

  public override mutate( options?: LineOptions ): this {
    return super.mutate( options );
  }

  /**
   * Returns available fill renderers. (scenery-internal)
   *
   * Since our line can't be filled, we support all fill renderers.
   *
   * See Renderer for more information on the bitmasks
   */
  public override getFillRendererBitmask(): number {
    return Renderer.bitmaskCanvas | Renderer.bitmaskSVG | Renderer.bitmaskDOM | Renderer.bitmaskWebGL;
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
Line.prototype._mutatorKeys = LINE_OPTION_KEYS.concat( Path.prototype._mutatorKeys );

/**
 * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
 *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
 *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
 * (scenery-internal)
 * @override
 */
Line.prototype.drawableMarkFlags = Path.prototype.drawableMarkFlags.concat( [ 'line', 'p1', 'p2', 'x1', 'x2', 'y1', 'y2' ] ).filter( flag => flag !== 'shape' );

scenery.register( 'Line', Line );
