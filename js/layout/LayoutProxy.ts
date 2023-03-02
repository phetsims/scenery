// Copyright 2022-2023, University of Colorado Boulder

/**
 * A stand-in for the layout-based fields of a Node, but where everything is done in the coordinate frame of the
 * "root" of the Trail. It is a pooled object, so it can be reused to avoid memory issues.
 *
 * NOTE: For layout, these trails usually have the "root" Node equal to the children of the layout constraint's ancestor
 * Node. Therefore, the coordinate space is typically the local coordinate frame of the ancestorNode of the
 * LayoutConstraint. This is not the same as the "global" coordinates for a Scenery Node in general (as most of the root
 * nodes of the trails provided to LayoutProxy will have parents!)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import Pool from '../../../phet-core/js/Pool.js';
import { HeightSizableNode, isHeightSizable, isWidthSizable, Node, scenery, Trail, WidthSizableNode } from '../imports.js';

export default class LayoutProxy {

  // Nulled out when disposed
  public trail!: Trail | null;

  /**
   * @param trail - The wrapped Node is the leaf-most node, but coordinates will be handled in the global frame
   * of the trail itself.
   */
  public constructor( trail: Trail ) {
    this.initialize( trail );
  }

  /**
   * This is where the logic of a poolable type's "initializer" will go. It will be potentially run MULTIPLE times,
   * as if it was constructing multiple different objects. It will be put back in the pool with dispose().
   * It will go through cycles of:
   * - constructor(...) => initialize(...) --- only at the start
   * - dispose()
   * - initialize(...)
   * - dispose()
   * - initialize(...)
   * - dispose()
   * and so on.
   *
   * DO not call it twice without in-between disposals (follow the above pattern).
   */
  public initialize( trail: Trail ): this {
    this.trail = trail;

    return this;
  }

  private checkPreconditions(): void {
    assert && assert( this.trail, 'Should not be disposed' );
    assert && assert( this.trail!.getParentMatrix().isAxisAligned(), 'Transforms with LayoutProxy need to be axis-aligned' );
  }

  public get node(): Node {
    assert && this.checkPreconditions();

    return this.trail!.lastNode();
  }

  /**
   * Returns the bounds of the last node in the trail, but in the root coordinate frame.
   */
  public get bounds(): Bounds2 {
    assert && this.checkPreconditions();

    return this.trail!.parentToGlobalBounds( this.node.bounds );
  }

  /**
   * Returns the visibleBounds of the last node in the trail, but in the root coordinate frame.
   */
  public get visibleBounds(): Bounds2 {
    assert && this.checkPreconditions();

    return this.trail!.parentToGlobalBounds( this.node.visibleBounds );
  }

  /**
   * Returns the width of the last node in the trail, but in the root coordinate frame.
   */
  public get width(): number {
    return this.bounds.width;
  }

  /**
   * Returns the height of the last node in the trail, but in the root coordinate frame.
   */
  public get height(): number {
    return this.bounds.height;
  }

  /**
   * Returns the x of the last node in the trail, but in the root coordinate frame.
   */
  public get x(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformX( this.node.x );
  }

  /**
   * Sets the x of the last node in the trail, but in the root coordinate frame.
   */
  public set x( value: number ) {
    assert && this.checkPreconditions();

    this.node.x = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the y of the last node in the trail, but in the root coordinate frame.
   */
  public get y(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformY( this.node.y );
  }

  /**
   * Sets the y of the last node in the trail, but in the root coordinate frame.
   */
  public set y( value: number ) {
    assert && this.checkPreconditions();

    this.node.y = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the translation of the last node in the trail, but in the root coordinate frame.
   */
  public get translation(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.translation );
  }

  /**
   * Sets the translation of the last node in the trail, but in the root coordinate frame.
   */
  public set translation( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.translation = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the left of the last node in the trail, but in the root coordinate frame.
   */
  public get left(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformX( this.node.left );
  }

  /**
   * Sets the left of the last node in the trail, but in the root coordinate frame.
   */
  public set left( value: number ) {
    this.node.left = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the right of the last node in the trail, but in the root coordinate frame.
   */
  public get right(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformX( this.node.right );
  }

  /**
   * Sets the right of the last node in the trail, but in the root coordinate frame.
   */
  public set right( value: number ) {
    assert && this.checkPreconditions();

    this.node.right = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the centerX of the last node in the trail, but in the root coordinate frame.
   */
  public get centerX(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformX( this.node.centerX );
  }

  /**
   * Sets the centerX of the last node in the trail, but in the root coordinate frame.
   */
  public set centerX( value: number ) {
    assert && this.checkPreconditions();

    this.node.centerX = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the top of the last node in the trail, but in the root coordinate frame.
   */
  public get top(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformY( this.node.top );
  }

  /**
   * Sets the top of the last node in the trail, but in the root coordinate frame.
   */
  public set top( value: number ) {
    assert && this.checkPreconditions();

    this.node.top = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the bottom of the last node in the trail, but in the root coordinate frame.
   */
  public get bottom(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformY( this.node.bottom );
  }

  /**
   * Sets the bottom of the last node in the trail, but in the root coordinate frame.
   */
  public set bottom( value: number ) {
    assert && this.checkPreconditions();

    this.node.bottom = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the centerY of the last node in the trail, but in the root coordinate frame.
   */
  public get centerY(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformY( this.node.centerY );
  }

  /**
   * Sets the centerY of the last node in the trail, but in the root coordinate frame.
   */
  public set centerY( value: number ) {
    assert && this.checkPreconditions();

    this.node.centerY = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the leftTop of the last node in the trail, but in the root coordinate frame.
   */
  public get leftTop(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.leftTop );
  }

  /**
   * Sets the leftTop of the last node in the trail, but in the root coordinate frame.
   */
  public set leftTop( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.leftTop = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the centerTop of the last node in the trail, but in the root coordinate frame.
   */
  public get centerTop(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.centerTop );
  }

  /**
   * Sets the centerTop of the last node in the trail, but in the root coordinate frame.
   */
  public set centerTop( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.centerTop = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightTop of the last node in the trail, but in the root coordinate frame.
   */
  public get rightTop(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.rightTop );
  }

  /**
   * Sets the rightTop of the last node in the trail, but in the root coordinate frame.
   */
  public set rightTop( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.rightTop = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the leftCenter of the last node in the trail, but in the root coordinate frame.
   */
  public get leftCenter(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.leftCenter );
  }

  /**
   * Sets the leftCenter of the last node in the trail, but in the root coordinate frame.
   */
  public set leftCenter( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.leftCenter = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the center of the last node in the trail, but in the root coordinate frame.
   */
  public get center(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.center );
  }

  /**
   * Sets the center of the last node in the trail, but in the root coordinate frame.
   */
  public set center( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.center = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightCenter of the last node in the trail, but in the root coordinate frame.
   */
  public get rightCenter(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.rightCenter );
  }

  /**
   * Sets the rightCenter of the last node in the trail, but in the root coordinate frame.
   */
  public set rightCenter( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.rightCenter = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the leftBottom of the last node in the trail, but in the root coordinate frame.
   */
  public get leftBottom(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.leftBottom );
  }

  /**
   * Sets the leftBottom of the last node in the trail, but in the root coordinate frame.
   */
  public set leftBottom( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.leftBottom = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the centerBottom of the last node in the trail, but in the root coordinate frame.
   */
  public get centerBottom(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.centerBottom );
  }

  /**
   * Sets the centerBottom of the last node in the trail, but in the root coordinate frame.
   */
  public set centerBottom( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.centerBottom = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightBottom of the last node in the trail, but in the root coordinate frame.
   */
  public get rightBottom(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.rightBottom );
  }

  /**
   * Sets the rightBottom of the last node in the trail, but in the root coordinate frame.
   */
  public set rightBottom( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.rightBottom = this.trail!.getParentTransform().inversePosition2( value );
  }

  public get widthSizable(): boolean {
    return this.node.widthSizable;
  }

  public get heightSizable(): boolean {
    return this.node.heightSizable;
  }

  public get preferredWidth(): number | null {
    assert && this.checkPreconditions();
    assert && assert( isWidthSizable( this.node ) );

    const preferredWidth = ( this.node as WidthSizableNode ).preferredWidth;

    return preferredWidth === null ? null : Math.abs( this.trail!.getParentTransform().transformDeltaX( preferredWidth ) );
  }

  public set preferredWidth( preferredWidth: number | null ) {
    assert && this.checkPreconditions();
    assert && assert( isWidthSizable( this.node ) );

    ( this.node as WidthSizableNode ).preferredWidth = preferredWidth === null ? null : Math.abs( this.trail!.getParentTransform().inverseDeltaX( preferredWidth ) );
  }

  public get preferredHeight(): number | null {
    assert && this.checkPreconditions();
    assert && assert( isHeightSizable( this.node ) );

    const preferredHeight = ( this.node as HeightSizableNode ).preferredHeight;

    return preferredHeight === null ? null : Math.abs( this.trail!.getParentTransform().transformDeltaY( preferredHeight ) );
  }

  public set preferredHeight( preferredHeight: number | null ) {
    assert && this.checkPreconditions();
    assert && assert( isHeightSizable( this.node ) );

    ( this.node as HeightSizableNode ).preferredHeight = preferredHeight === null ? null : Math.abs( this.trail!.getParentTransform().inverseDeltaY( preferredHeight ) );
  }

  public get minimumWidth(): number {
    assert && this.checkPreconditions();

    const minimumWidth = isWidthSizable( this.node ) ? this.node.minimumWidth || 0 : this.node.width;

    return Math.abs( this.trail!.getParentTransform().transformDeltaX( minimumWidth ) );
  }

  public get minimumHeight(): number {
    assert && this.checkPreconditions();

    const minimumHeight = isHeightSizable( this.node ) ? this.node.minimumHeight || 0 : this.node.height;

    return Math.abs( this.trail!.getParentTransform().transformDeltaY( minimumHeight ) );
  }

  public getMinimum( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.minimumWidth : this.minimumHeight;
  }

  public get maxWidth(): number | null {
    assert && this.checkPreconditions();

    if ( this.node.maxWidth === null ) {
      return null;
    }
    else {
      return Math.abs( this.trail!.getParentTransform().transformDeltaX( this.node.maxWidth ) );
    }
  }

  public set maxWidth( value: number | null ) {
    assert && this.checkPreconditions();

    this.node.maxWidth = value === null ? null : Math.abs( this.trail!.getParentTransform().inverseDeltaX( value ) );
  }

  public get maxHeight(): number | null {
    assert && this.checkPreconditions();

    if ( this.node.maxHeight === null ) {
      return null;
    }
    else {
      return Math.abs( this.trail!.getParentTransform().transformDeltaY( this.node.maxHeight ) );
    }
  }

  public set maxHeight( value: number | null ) {
    assert && this.checkPreconditions();

    this.node.maxHeight = value === null ? null : Math.abs( this.trail!.getParentTransform().inverseDeltaY( value ) );
  }

  /**
   * Returns either the maxWidth or maxHeight depending on the orientation
   */
  public getMax( orientation: Orientation ): number | null {
    return orientation === Orientation.HORIZONTAL ? this.maxWidth : this.maxHeight;
  }

  public get visible(): boolean {
    return this.node.visible;
  }

  public set visible( value: boolean ) {
    this.node.visible = value;
  }

  /**
   * Releases references, and frees it to the pool.
   */
  public dispose(): void {
    this.trail = null;

    this.freeToPool();
  }

  public freeToPool(): void {
    LayoutProxy.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( LayoutProxy, {
    maxSize: 1000
  } );
}

scenery.register( 'LayoutProxy', LayoutProxy );
