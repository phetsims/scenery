// Copyright 2022, University of Colorado Boulder

/**
 * A stand-in for the layout-based fields of a Node, but where everything is done in the coordinate frame of the
 * "root" of the Trail.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Pool from '../../../phet-core/js/Pool.js';
import { HeightSizableNode, isHeightSizable, isWidthSizable, Node, scenery, Trail, WidthSizableNode } from '../imports.js';

export default class LayoutProxy {

  // Nulled out when disposed
  trail!: Trail | null;

  /**
   * @param  trail - The wrapped Node is the leaf-most node, but coordinates will be handled in the global frame
   * of the trail itself.
   */
  constructor( trail: Trail ) {
    this.initialize( trail );
  }

  initialize( trail: Trail ): void {
    this.trail = trail;
  }

  checkPreconditions(): void {
    assert && assert( this.trail, 'Should not be disposed' );
    assert && assert( this.trail!.getParentMatrix().isAxisAligned(), 'Transforms with LayoutProxy need to be axis-aligned' );
  }

  get node(): Node {
    assert && this.checkPreconditions();

    return this.trail!.lastNode();
  }

  /**
   * Returns the bounds of the last node in the trail, but in the root coordinate frame.
   */
  get bounds(): Bounds2 {
    assert && this.checkPreconditions();

    // TODO: optimizations if we're the only node in the Trail!!
    return this.trail!.parentToGlobalBounds( this.node.bounds );
  }

  /**
   * Returns the visibleBounds of the last node in the trail, but in the root coordinate frame.
   */
  get visibleBounds(): Bounds2 {
    assert && this.checkPreconditions();

    return this.trail!.parentToGlobalBounds( this.node.visibleBounds );
  }

  /**
   * Returns the width of the last node in the trail, but in the root coordinate frame.
   */
  get width(): number {
    return this.bounds.width;
  }

  /**
   * Returns the height of the last node in the trail, but in the root coordinate frame.
   */
  get height(): number {
    return this.bounds.height;
  }

  /**
   * Returns the x of the last node in the trail, but in the root coordinate frame.
   */
  get x(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformX( this.node.x );
  }

  /**
   * Sets the x of the last node in the trail, but in the root coordinate frame.
   */
  set x( value: number ) {
    assert && this.checkPreconditions();

    this.node.x = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the y of the last node in the trail, but in the root coordinate frame.
   */
  get y(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformY( this.node.y );
  }

  /**
   * Sets the y of the last node in the trail, but in the root coordinate frame.
   */
  set y( value: number ) {
    assert && this.checkPreconditions();

    this.node.y = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the translation of the last node in the trail, but in the root coordinate frame.
   */
  get translation(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.translation );
  }

  /**
   * Sets the translation of the last node in the trail, but in the root coordinate frame.
   */
  set translation( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.translation = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the left of the last node in the trail, but in the root coordinate frame.
   */
  get left(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformX( this.node.left );
  }

  /**
   * Sets the left of the last node in the trail, but in the root coordinate frame.
   */
  set left( value: number ) {
    this.node.left = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the right of the last node in the trail, but in the root coordinate frame.
   */
  get right(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformX( this.node.right );
  }

  /**
   * Sets the right of the last node in the trail, but in the root coordinate frame.
   */
  set right( value: number ) {
    assert && this.checkPreconditions();

    this.node.right = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the centerX of the last node in the trail, but in the root coordinate frame.
   */
  get centerX(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformX( this.node.centerX );
  }

  /**
   * Sets the centerX of the last node in the trail, but in the root coordinate frame.
   */
  set centerX( value: number ) {
    assert && this.checkPreconditions();

    this.node.centerX = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the top of the last node in the trail, but in the root coordinate frame.
   */
  get top(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformY( this.node.top );
  }

  /**
   * Sets the top of the last node in the trail, but in the root coordinate frame.
   */
  set top( value: number ) {
    assert && this.checkPreconditions();

    this.node.top = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the bottom of the last node in the trail, but in the root coordinate frame.
   */
  get bottom(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformY( this.node.bottom );
  }

  /**
   * Sets the bottom of the last node in the trail, but in the root coordinate frame.
   */
  set bottom( value: number ) {
    assert && this.checkPreconditions();

    this.node.bottom = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the centerY of the last node in the trail, but in the root coordinate frame.
   */
  get centerY(): number {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformY( this.node.centerY );
  }

  /**
   * Sets the centerY of the last node in the trail, but in the root coordinate frame.
   */
  set centerY( value: number ) {
    assert && this.checkPreconditions();

    this.node.centerY = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the leftTop of the last node in the trail, but in the root coordinate frame.
   */
  get leftTop(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.leftTop );
  }

  /**
   * Sets the leftTop of the last node in the trail, but in the root coordinate frame.
   */
  set leftTop( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.leftTop = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the centerTop of the last node in the trail, but in the root coordinate frame.
   */
  get centerTop(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.centerTop );
  }

  /**
   * Sets the centerTop of the last node in the trail, but in the root coordinate frame.
   */
  set centerTop( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.centerTop = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightTop of the last node in the trail, but in the root coordinate frame.
   */
  get rightTop(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.rightTop );
  }

  /**
   * Sets the rightTop of the last node in the trail, but in the root coordinate frame.
   */
  set rightTop( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.rightTop = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the leftCenter of the last node in the trail, but in the root coordinate frame.
   */
  get leftCenter(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.leftCenter );
  }

  /**
   * Sets the leftCenter of the last node in the trail, but in the root coordinate frame.
   */
  set leftCenter( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.leftCenter = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the center of the last node in the trail, but in the root coordinate frame.
   */
  get center(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.center );
  }

  /**
   * Sets the center of the last node in the trail, but in the root coordinate frame.
   */
  set center( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.center = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightCenter of the last node in the trail, but in the root coordinate frame.
   */
  get rightCenter(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.rightCenter );
  }

  /**
   * Sets the rightCenter of the last node in the trail, but in the root coordinate frame.
   */
  set rightCenter( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.rightCenter = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the leftBottom of the last node in the trail, but in the root coordinate frame.
   */
  get leftBottom(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.leftBottom );
  }

  /**
   * Sets the leftBottom of the last node in the trail, but in the root coordinate frame.
   */
  set leftBottom( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.leftBottom = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the centerBottom of the last node in the trail, but in the root coordinate frame.
   */
  get centerBottom(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.centerBottom );
  }

  /**
   * Sets the centerBottom of the last node in the trail, but in the root coordinate frame.
   */
  set centerBottom( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.centerBottom = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightBottom of the last node in the trail, but in the root coordinate frame.
   */
  get rightBottom(): Vector2 {
    assert && this.checkPreconditions();

    return this.trail!.getParentTransform().transformPosition2( this.node.rightBottom );
  }

  /**
   * Sets the rightBottom of the last node in the trail, but in the root coordinate frame.
   */
  set rightBottom( value: Vector2 ) {
    assert && this.checkPreconditions();

    this.node.rightBottom = this.trail!.getParentTransform().inversePosition2( value );
  }

  get widthSizable(): boolean {
    return this.node.widthSizable;
  }

  get heightSizable(): boolean {
    return this.node.heightSizable;
  }

  get preferredWidth(): number | null {
    assert && this.checkPreconditions();
    assert && assert( isWidthSizable( this.node ) );

    const preferredWidth = ( this.node as WidthSizableNode ).preferredWidth;

    return preferredWidth === null ? null : Math.abs( this.trail!.getParentTransform().transformDeltaX( preferredWidth ) );
  }

  get preferredHeight(): number | null {
    assert && this.checkPreconditions();
    assert && assert( isHeightSizable( this.node ) );

    const preferredHeight = ( this.node as HeightSizableNode ).preferredHeight;

    return preferredHeight === null ? null : Math.abs( this.trail!.getParentTransform().transformDeltaY( preferredHeight ) );
  }

  set preferredWidth( preferredWidth: number | null ) {
    assert && this.checkPreconditions();
    assert && assert( isWidthSizable( this.node ) );

    ( this.node as WidthSizableNode ).preferredWidth = preferredWidth === null ? null : Math.abs( this.trail!.getParentTransform().inverseDeltaX( preferredWidth ) );
  }

  set preferredHeight( preferredHeight: number | null ) {
    assert && this.checkPreconditions();
    assert && assert( isHeightSizable( this.node ) );

    ( this.node as HeightSizableNode ).preferredHeight = preferredHeight === null ? null : Math.abs( this.trail!.getParentTransform().inverseDeltaY( preferredHeight ) );
  }

  get minimumWidth(): number {
    assert && this.checkPreconditions();

    const minimumWidth = isWidthSizable( this.node ) ? this.node.minimumWidth || 0 : this.node.width;

    return Math.abs( this.trail!.getParentTransform().transformDeltaX( minimumWidth ) );
  }

  get minimumHeight(): number {
    assert && this.checkPreconditions();

    const minimumHeight = isHeightSizable( this.node ) ? this.node.minimumHeight || 0 : this.node.height;

    return Math.abs( this.trail!.getParentTransform().transformDeltaY( minimumHeight ) );
  }

  /**
   * Releases references, and frees it to the pool.
   */
  dispose(): void {
    this.trail = null;

    this.freeToPool();
  }

  freeToPool(): void {
    LayoutProxy.pool.freeToPool( this );
  }

  static readonly pool = new Pool( LayoutProxy, {
    maxSize: 1000
  } );
}

scenery.register( 'LayoutProxy', LayoutProxy );
