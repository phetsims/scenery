// Copyright 2022, University of Colorado Boulder

/**
 * A stand-in for the layout-based fields of a Node, but where everything is done in the coordinate frame of the
 * "root" of the Trail.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Poolable, { PoolableVersion } from '../../../phet-core/js/Poolable.js';
import { scenery, Trail, Node } from '../imports.js';

class LayoutProxy {

  // Nulled out when disposed
  trail!: Trail | null;

  /**
   * @param  trail - The wrapped Node is the leaf-most node, but coordinates will be handled in the global frame
   * of the trail itself.
   */
  constructor( trail: Trail ) {
    this.initialize( trail );
  }

  initialize( trail: Trail ) {
    this.trail = trail;
  }

  get node(): Node {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.lastNode();
  }

  /**
   * Returns the bounds of the last node in the trail, but in the root coordinate frame.
   */
  get bounds(): Bounds2 {
    assert && assert( this.trail, 'Should not be disposed' );

    // TODO: optimizations if we're the only node in the Trail!!
    return this.trail!.parentToGlobalBounds( this.node.bounds );
  }

  /**
   * Returns the visibleBounds of the last node in the trail, but in the root coordinate frame.
   */
  get visibleBounds(): Bounds2 {
    assert && assert( this.trail, 'Should not be disposed' );

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
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformX( this.node.x );
  }

  /**
   * Sets the x of the last node in the trail, but in the root coordinate frame.
   */
  set x( value: number ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.x = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the y of the last node in the trail, but in the root coordinate frame.
   */
  get y(): number {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformY( this.node.y );
  }

  /**
   * Sets the y of the last node in the trail, but in the root coordinate frame.
   */
  set y( value: number ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.y = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the translation of the last node in the trail, but in the root coordinate frame.
   */
  get translation(): Vector2 {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformPosition2( this.node.translation );
  }

  /**
   * Sets the translation of the last node in the trail, but in the root coordinate frame.
   */
  set translation( value: Vector2 ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.translation = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the left of the last node in the trail, but in the root coordinate frame.
   */
  get left(): number {
    assert && assert( this.trail, 'Should not be disposed' );

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
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformX( this.node.right );
  }

  /**
   * Sets the right of the last node in the trail, but in the root coordinate frame.
   */
  set right( value: number ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.right = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the centerX of the last node in the trail, but in the root coordinate frame.
   */
  get centerX(): number {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformX( this.node.centerX );
  }

  /**
   * Sets the centerX of the last node in the trail, but in the root coordinate frame.
   */
  set centerX( value: number ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.centerX = this.trail!.getParentTransform().inverseX( value );
  }

  /**
   * Returns the top of the last node in the trail, but in the root coordinate frame.
   */
  get top(): number {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformY( this.node.top );
  }

  /**
   * Sets the top of the last node in the trail, but in the root coordinate frame.
   */
  set top( value: number ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.top = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the bottom of the last node in the trail, but in the root coordinate frame.
   */
  get bottom(): number {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformY( this.node.bottom );
  }

  /**
   * Sets the bottom of the last node in the trail, but in the root coordinate frame.
   */
  set bottom( value: number ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.bottom = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the centerY of the last node in the trail, but in the root coordinate frame.
   */
  get centerY(): number {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformY( this.node.centerY );
  }

  /**
   * Sets the centerY of the last node in the trail, but in the root coordinate frame.
   */
  set centerY( value: number ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.centerY = this.trail!.getParentTransform().inverseY( value );
  }

  /**
   * Returns the leftTop of the last node in the trail, but in the root coordinate frame.
   */
  get leftTop(): Vector2 {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformPosition2( this.node.leftTop );
  }

  /**
   * Sets the leftTop of the last node in the trail, but in the root coordinate frame.
   */
  set leftTop( value: Vector2 ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.leftTop = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the centerTop of the last node in the trail, but in the root coordinate frame.
   */
  get centerTop(): Vector2 {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformPosition2( this.node.centerTop );
  }

  /**
   * Sets the centerTop of the last node in the trail, but in the root coordinate frame.
   */
  set centerTop( value: Vector2 ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.centerTop = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightTop of the last node in the trail, but in the root coordinate frame.
   */
  get rightTop(): Vector2 {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformPosition2( this.node.rightTop );
  }

  /**
   * Sets the rightTop of the last node in the trail, but in the root coordinate frame.
   */
  set rightTop( value: Vector2 ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.rightTop = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the leftCenter of the last node in the trail, but in the root coordinate frame.
   */
  get leftCenter(): Vector2 {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformPosition2( this.node.leftCenter );
  }

  /**
   * Sets the leftCenter of the last node in the trail, but in the root coordinate frame.
   */
  set leftCenter( value: Vector2 ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.leftCenter = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the center of the last node in the trail, but in the root coordinate frame.
   */
  get center(): Vector2 {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformPosition2( this.node.center );
  }

  /**
   * Sets the center of the last node in the trail, but in the root coordinate frame.
   */
  set center( value: Vector2 ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.center = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightCenter of the last node in the trail, but in the root coordinate frame.
   */
  get rightCenter(): Vector2 {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformPosition2( this.node.rightCenter );
  }

  /**
   * Sets the rightCenter of the last node in the trail, but in the root coordinate frame.
   */
  set rightCenter( value: Vector2 ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.rightCenter = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the leftBottom of the last node in the trail, but in the root coordinate frame.
   */
  get leftBottom(): Vector2 {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformPosition2( this.node.leftBottom );
  }

  /**
   * Sets the leftBottom of the last node in the trail, but in the root coordinate frame.
   */
  set leftBottom( value: Vector2 ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.leftBottom = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the centerBottom of the last node in the trail, but in the root coordinate frame.
   */
  get centerBottom(): Vector2 {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformPosition2( this.node.centerBottom );
  }

  /**
   * Sets the centerBottom of the last node in the trail, but in the root coordinate frame.
   */
  set centerBottom( value: Vector2 ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.centerBottom = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightBottom of the last node in the trail, but in the root coordinate frame.
   */
  get rightBottom(): Vector2 {
    assert && assert( this.trail, 'Should not be disposed' );

    return this.trail!.getParentTransform().transformPosition2( this.node.rightBottom );
  }

  /**
   * Sets the rightBottom of the last node in the trail, but in the root coordinate frame.
   */
  set rightBottom( value: Vector2 ) {
    assert && assert( this.trail, 'Should not be disposed' );

    this.node.rightBottom = this.trail!.getParentTransform().inversePosition2( value );
  }

  /**
   * Releases references, and frees it to the pool.
   * @public
   */
  dispose() {
    this.trail = null;

    // for now
    ( this as unknown as PoolableLayoutProxy ).freeToPool();
  }
}

scenery.register( 'LayoutProxy', LayoutProxy );

type PoolableLayoutProxy = PoolableVersion<typeof LayoutProxy>;
const PoolableLayoutProxy = Poolable.mixInto( LayoutProxy, { // eslint-disable-line
  maxSize: 1000
} );

export default PoolableLayoutProxy;