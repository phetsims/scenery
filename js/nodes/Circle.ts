// Copyright 2013-2023, University of Colorado Boulder

/**
 * A circular node that inherits Path, and allows for optimized drawing and improved parameter handling.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import extendDefined from '../../../phet-core/js/extendDefined.js';
import { CanvasContextWrapper, CanvasSelfDrawable, CircleCanvasDrawable, CircleDOMDrawable, CircleSVGDrawable, DOMSelfDrawable, Features, TCircleDrawable, Instance, Path, PathOptions, Renderer, scenery, SVGSelfDrawable, VoicingOptions } from '../imports.js';

const CIRCLE_OPTION_KEYS = [
  'radius' // {number} - see setRadius() for more documentation
];

type SelfOptions = {
  radius?: number;
};

export type CircleOptions = SelfOptions & VoicingOptions & StrictOmit<PathOptions, 'shape'>;

export default class Circle extends Path {

  // The radius of the circle
  private _radius: number;

  /**
   * NOTE: There are two ways of invoking the constructor:
   * - new Circle( radius, { ... } )
   * - new Circle( { radius: radius, ... } )
   *
   * This allows the radius to be included in the parameter object for when that is convenient.
   *
   * @param radius - The (non-negative) radius of the circle
   * @param  [options] - Circle-specific options are documented in CIRCLE_OPTION_KEYS above, and can be provided
   *                     along-side options for Node
   */
  public constructor( options?: CircleOptions );
  public constructor( radius: number, options?: CircleOptions );
  public constructor( radius?: number | CircleOptions, options?: CircleOptions ) {
    super( null );

    this._radius = 0;

    // Handle new Circle( { radius: ... } )
    if ( typeof radius === 'object' ) {
      options = radius;
      assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
        'Extra prototype on Node options object is a code smell' );
    }
    // Handle new Circle( radius, { ... } )
    else {
      assert && assert( options === undefined || Object.getPrototypeOf( options ) === Object.prototype,
        'Extra prototype on Node options object is a code smell' );
      options = extendDefined( {
        radius: radius
      }, options );
    }

    this.mutate( options );
  }


  /**
   * Determines the default allowed renderers (returned via the Renderer bitmask) that are allowed, given the
   * current stroke options. (scenery-internal)
   *
   * We can support the DOM renderer if there is a solid-styled stroke (which otherwise wouldn't be supported).
   */
  public override getStrokeRendererBitmask(): number {
    let bitmask = super.getStrokeRendererBitmask();
    // @ts-expect-error TODO isGradient/isPattern better handling
    if ( this.hasStroke() && !this.getStroke()!.isGradient && !this.getStroke()!.isPattern && this.getLineWidth() <= this.getRadius() ) {
      bitmask |= Renderer.bitmaskDOM;
    }
    return bitmask;
  }

  /**
   * Determines the allowed renderers that are allowed (or excluded) based on the current Path. (scenery-internal)
   */
  public override getPathRendererBitmask(): number {
    // If we can use CSS borderRadius, we can support the DOM renderer.
    return Renderer.bitmaskCanvas | Renderer.bitmaskSVG | ( Features.borderRadius ? Renderer.bitmaskDOM : 0 );
  }

  /**
   * Notifies that the circle has changed (probably the radius), and invalidates path information and our cached
   * shape.
   */
  private invalidateCircle(): void {
    assert && assert( this._radius >= 0, 'A circle needs a non-negative radius' );

    // sets our 'cache' to null, so we don't always have to recompute our shape
    this._shape = null;

    // should invalidate the path and ensure a redraw
    this.invalidatePath();
  }

  /**
   * Returns a Shape that is equivalent to our rendered display. Generally used to lazily create a Shape instance
   * when one is needed, without having to do so beforehand.
   */
  private createCircleShape(): Shape {
    return Shape.circle( 0, 0, this._radius ).makeImmutable();
  }

  /**
   * Returns whether this Circle's selfBounds is intersected by the specified bounds.
   *
   * @param bounds - Bounds to test, assumed to be in the local coordinate frame.
   */
  public override intersectsBoundsSelf( bounds: Bounds2 ): boolean {
    // TODO: handle intersection with somewhat-infinite bounds!
    let x = Math.abs( bounds.centerX );
    let y = Math.abs( bounds.centerY );
    const halfWidth = bounds.maxX - x;
    const halfHeight = bounds.maxY - y;

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
    CircleCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
  }

  /**
   * Creates a DOM drawable for this Circle. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createDOMDrawable( renderer: number, instance: Instance ): DOMSelfDrawable {
    // @ts-expect-error TODO: pooling
    return CircleDOMDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a SVG drawable for this Circle. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createSVGDrawable( renderer: number, instance: Instance ): SVGSelfDrawable {
    // @ts-expect-error TODO: pooling
    return CircleSVGDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a Canvas drawable for this Circle. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createCanvasDrawable( renderer: number, instance: Instance ): CanvasSelfDrawable {
    // @ts-expect-error TODO: pooling
    return CircleCanvasDrawable.createFromPool( renderer, instance );
  }

  /**
   * Sets the radius of the circle.
   */
  public setRadius( radius: number ): this {
    assert && assert( radius >= 0, 'A circle needs a non-negative radius' );
    assert && assert( isFinite( radius ), 'A circle needs a finite radius' );

    if ( this._radius !== radius ) {
      this._radius = radius;
      this.invalidateCircle();

      const stateLen = this._drawables.length;
      for ( let i = 0; i < stateLen; i++ ) {
        ( this._drawables[ i ] as unknown as TCircleDrawable ).markDirtyRadius();
      }
    }
    return this;
  }

  public set radius( value: number ) { this.setRadius( value ); }

  public get radius(): number { return this.getRadius(); }

  /**
   * Returns the radius of the circle.
   */
  public getRadius(): number {
    return this._radius;
  }

  /**
   * Computes the bounds of the Circle, including any applied stroke. Overridden for efficiency.
   */
  public override computeShapeBounds(): Bounds2 {
    let bounds = new Bounds2( -this._radius, -this._radius, this._radius, this._radius );
    if ( this._stroke ) {
      // since we are axis-aligned, any stroke will expand our bounds by a guaranteed set amount
      bounds = bounds.dilated( this.getLineWidth() / 2 );
    }
    return bounds;
  }

  /**
   * Computes whether the provided point is "inside" (contained) in this Circle's self content, or "outside".
   *
   * Exists to optimize hit detection, as it's quick to compute for circles.
   *
   * @param point - Considered to be in the local coordinate frame
   */
  public override containsPointSelf( point: Vector2 ): boolean {
    const magSq = point.x * point.x + point.y * point.y;
    let result = true;
    let iRadius: number;
    if ( this._strokePickable ) {
      iRadius = this.getLineWidth() / 2;
      const outerRadius = this._radius + iRadius;
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
      const innerRadius = this._radius - ( iRadius! );
      return result && magSq >= innerRadius * innerRadius;
    }
    else {
      return false; // neither stroke nor fill is pickable
    }
  }

  /**
   * It is impossible to set another shape on this Path subtype, as its effective shape is determined by other
   * parameters.
   *
   * @param shape - Throws an error if it is not null.
   */
  public override setShape( shape: Shape | null ): this {
    if ( shape !== null ) {
      throw new Error( 'Cannot set the shape of a Circle to something non-null' );
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
      this._shape = this.createCircleShape();
    }
    return this._shape;
  }

  /**
   * Returns whether this Path has an associated Shape (instead of no shape, represented by null)
   */
  public override hasShape(): boolean {
    // Always true for this Path subtype
    return true;
  }

  public override mutate( options?: CircleOptions ): this {
    return super.mutate( options );
  }

}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
Circle.prototype._mutatorKeys = CIRCLE_OPTION_KEYS.concat( Path.prototype._mutatorKeys );

/**
 * {Array.<String>} - List of all dirty flags that should be available on drawables created from this node (or
 *                    subtype). Given a flag (e.g. radius), it indicates the existence of a function
 *                    drawable.markDirtyRadius() that will indicate to the drawable that the radius has changed.
 * (scenery-internal)
 * @override
 */
Circle.prototype.drawableMarkFlags = Path.prototype.drawableMarkFlags.concat( [ 'radius' ] ).filter( flag => flag !== 'shape' );

scenery.register( 'Circle', Circle );
