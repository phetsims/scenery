// Copyright 2019-2023, University of Colorado Boulder

/**
 * Meant for displaying a large amount of high-performance instances of sprites.
 * See https://github.com/phetsims/scenery/issues/990 for more information.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import optionize from '../../../phet-core/js/optionize.js';
import { CanvasContextWrapper, CanvasSelfDrawable, Instance, Node, NodeOptions, Renderer, scenery, Sprite, SpriteInstance, SpritesCanvasDrawable, SpritesWebGLDrawable, WebGLSelfDrawable } from '../imports.js';

type SelfOptions = {
  // Provide a fixed set of Sprite objects that will be used for this node. Currently, it cannot be modified after
  // construction for simplicity of the implementation.
  sprites?: Sprite[];

  // A reference to an Array of instances. This array can be (and should be)
  // manipulated to adjust the displayed instances (their sprites/transforms/etc.). After this has been adjusted,
  // invalidatePaint() should be called on the Sprites node.
  spriteInstances?: SpriteInstance[];

  // Whether individual sprites will be hit-tested to determine what is a contained point (for hit
  // testing, etc.). If false, the canvasBounds will be used for hit testing.
  hitTestSprites?: boolean;

  canvasBounds?: Bounds2;
};

// We don't specify a default for canvasBounds on purpose, so we'll omit this from the optionize type parameter.
type SpecifiedSelfOptions = StrictOmit<SelfOptions, 'canvasBounds'>;

export type SpritesOptions = SelfOptions & NodeOptions;

export default class Sprites extends Node {

  private _sprites: Sprite[];
  private _spriteInstances: SpriteInstance[];
  private _hitTestSprites: boolean;

  public constructor( providedOptions?: SpritesOptions ) {

    const options = optionize<SpritesOptions, SpecifiedSelfOptions, NodeOptions>()( {
      sprites: [],
      spriteInstances: [],
      hitTestSprites: false,

      // Sets the node's default renderer to WebGL (as we'll generally want that when using this type)
      renderer: 'webgl'
    }, providedOptions );

    super();

    this._sprites = options.sprites;
    this._spriteInstances = options.spriteInstances;
    this._hitTestSprites = options.hitTestSprites;

    // WebGL and Canvas are supported.
    this.setRendererBitmask( Renderer.bitmaskCanvas | Renderer.bitmaskWebGL );

    this.mutate( options );
  }

  /**
   * Sets the bounds that are used for layout/repainting.
   *
   * These bounds should always cover at least the area where the Sprites will draw in. If this is violated, this
   * node may be partially or completely invisible in Scenery's output.
   */
  public setCanvasBounds( selfBounds: Bounds2 ): void {
    this.invalidateSelf( selfBounds );
  }

  public set canvasBounds( value: Bounds2 ) { this.setCanvasBounds( value ); }

  public get canvasBounds(): Bounds2 { return this.getCanvasBounds(); }

  /**
   * Returns the previously-set canvasBounds, or Bounds2.NOTHING if it has not been set yet.
   */
  public getCanvasBounds(): Bounds2 {
    return this.getSelfBounds();
  }

  /**
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   *
   * @param wrapper
   * @param matrix - The transformation matrix already applied to the context.
   */
  protected override canvasPaintSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ): void {
    SpritesCanvasDrawable.prototype.paintCanvas( wrapper, this, matrix );
  }

  /**
   * Creates a Canvas drawable for this Sprites node. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createCanvasDrawable( renderer: number, instance: Instance ): CanvasSelfDrawable {
    // @ts-expect-error Pooling
    return SpritesCanvasDrawable.createFromPool( renderer, instance );
  }

  /**
   * Creates a WebGL drawable for this Image. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createWebGLDrawable( renderer: number, instance: Instance ): WebGLSelfDrawable {
    // @ts-expect-error Pooling
    return SpritesWebGLDrawable.createFromPool( renderer, instance );
  }

  /**
   * Computes whether the provided point is "inside" (contained) in this Node's self content, or "outside".
   *
   * @param point - Considered to be in the local coordinate frame
   */
  public override containsPointSelf( point: Vector2 ): boolean {
    const inBounds = super.containsPointSelf( point );
    if ( !inBounds ) {
      return false;
    }

    if ( this._hitTestSprites ) {
      return !!this.getSpriteInstanceFromPoint( point );
    }
    else {
      return true;
    }
  }

  /**
   * Finds which sprite instance is on top under a certain point (or null if none are).
   */
  public getSpriteInstanceFromPoint( point: Vector2 ): SpriteInstance | null {
    for ( let i = this._spriteInstances.length - 1; i >= 0; i-- ) {
      if ( this._spriteInstances[ i ].containsPoint( point ) ) {
        return this._spriteInstances[ i ];
      }
    }
    return null;
  }

  /**
   * Returns a Shape that represents the area covered by containsPointSelf.
   */
  public override getSelfShape(): Shape {
    if ( this._hitTestSprites ) {
      return Shape.union( this._spriteInstances.map( instance => instance.getShape() ) );
    }
    else {
      return Shape.bounds( this.selfBounds );
    }
  }

  /**
   * Whether this Node itself is painted (displays something itself).
   */
  public override isPainted(): boolean {
    // Always true for Sprites nodes
    return true;
  }

  /**
   * Should be called when this node needs to be repainted. When not called, Scenery assumes that this node does
   * NOT need to be repainted (although Scenery may repaint it due to other nodes needing to be repainted).
   *
   * This sets a "dirty" flag, so that it will be repainted the next time it would be displayed.
   */
  public invalidatePaint(): void {
    const stateLen = this._drawables.length;
    for ( let i = 0; i < stateLen; i++ ) {
      this._drawables[ i ].markDirty();
    }
  }

  public override mutate( options?: SpritesOptions ): this {
    return super.mutate( options );
  }
}

/**
 * {Array.<string>} - String keys for all the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
Sprites.prototype._mutatorKeys = [ 'canvasBounds' ].concat( Node.prototype._mutatorKeys );

scenery.register( 'Sprites', Sprites );
