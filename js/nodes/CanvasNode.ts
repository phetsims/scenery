// Copyright 2013-2023, University of Colorado Boulder

/**
 * An abstract node (should be subtyped) that is drawn by user-provided custom Canvas code.
 *
 * The region that can be drawn in is handled manually, by controlling the canvasBounds property of this CanvasNode.
 * Any regions outside of the canvasBounds will not be guaranteed to be drawn. This can be set with canvasBounds in the
 * constructor, or later with node.canvasBounds = bounds or setCanvasBounds( bounds ).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import { CanvasContextWrapper, CanvasNodeDrawable, CanvasSelfDrawable, Instance, Node, NodeOptions, Renderer, scenery } from '../imports.js';

const CANVAS_NODE_OPTION_KEYS = [
  'canvasBounds'
];

type SelfOptions = {
  canvasBounds?: Bounds2;
};

export type CanvasNodeOptions = SelfOptions & NodeOptions;

export default abstract class CanvasNode extends Node {
  public constructor( options?: CanvasNodeOptions ) {
    super( options );

    // This shouldn't change, as we only support one renderer
    this.setRendererBitmask( Renderer.bitmaskCanvas );
  }

  /**
   * Sets the bounds that are used for layout/repainting.
   *
   * These bounds should always cover at least the area where the CanvasNode will draw in. If this is violated, this
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
   * Whether this Node itself is painted (displays something itself).
   */
  public override isPainted(): boolean {
    // Always true for CanvasNode
    return true;
  }

  /**
   * Override paintCanvas with a faster version, since fillRect and drawRect don't affect the current default path.
   *
   * IMPORTANT NOTE: This function will be run from inside Scenery's Display.updateDisplay(), so it should not modify
   * or mutate any Scenery nodes (particularly anything that would cause something to be marked as needing a repaint).
   * Ideally, this function should have no outside effects other than painting to the Canvas provided.
   */
  public abstract paintCanvas( context: CanvasRenderingContext2D ): void;

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

  /**
   * Draws the current Node's self representation, assuming the wrapper's Canvas context is already in the local
   * coordinate frame of this node.
   *
   * @param wrapper
   * @param matrix - The transformation matrix already applied to the context.
   */
  protected override canvasPaintSelf( wrapper: CanvasContextWrapper, matrix: Matrix3 ): void {
    this.paintCanvas( wrapper.context );
  }

  /**
   * Computes whether the provided point is "inside" (contained) in this Node's self content, or "outside".
   *
   * If CanvasNode subtypes want to support being picked or hit-tested, it should override this function.
   *
   * @param point - Considered to be in the local coordinate frame
   */
  public override containsPointSelf( point: Vector2 ): boolean {
    return false;
  }

  /**
   * Returns a Shape that represents the area covered by containsPointSelf.
   */
  public override getSelfShape(): Shape {
    return new Shape();
  }

  /**
   * Creates a Canvas drawable for this CanvasNode. (scenery-internal)
   *
   * @param renderer - In the bitmask format specified by Renderer, which may contain additional bit flags.
   * @param instance - Instance object that will be associated with the drawable
   */
  public override createCanvasDrawable( renderer: number, instance: Instance ): CanvasSelfDrawable {
    // @ts-expect-error
    return CanvasNodeDrawable.createFromPool( renderer, instance );
  }

  public override mutate( options?: CanvasNodeOptions ): this {
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
CanvasNode.prototype._mutatorKeys = CANVAS_NODE_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

scenery.register( 'CanvasNode', CanvasNode );
