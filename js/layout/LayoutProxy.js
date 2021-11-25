// Copyright 2021, University of Colorado Boulder

/**
 * A stand-in for the layout-based fields of a Node, but where everything is done in the coordinate frame of the
 * "root" of the Trail.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../phet-core/js/Poolable.js';
import { scenery } from '../imports.js';

class LayoutProxy {
  /**
   * @mixes Poolable
   *
   * @param {Trail} trail - The wrapped Node is the leaf-most node, but coordinates will be handled in the global frame
   * of the trail itself.
   */
  constructor( trail ) {
    this.initialize( trail );
  }

  /**
   * @public
   *
   * @param {Trail} trail
   */
  initialize( trail ) {
    // @public {Trail|null} - Nulled out when disposed
    this.trail = trail;
  }

  /**
   * @public
   *
   * @returns {Node}
   */
  get node() {
    return this.trail.lastNode();
  }

  /**
   * Returns the bounds of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Bounds2} - In the root coordinate frame
   */
  get bounds() {
    // TODO: optimizations if we're the only node in the Trail!!
    return this.trail.parentToGlobalBounds( this.node.bounds );
  }

  /**
   * Returns the visibleBounds of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Bounds2} - In the root coordinate frame
   */
  get visibleBounds() {
    return this.trail.parentToGlobalBounds( this.node.visibleBounds );
  }

  /**
   * Returns the width of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {number} - In the root coordinate frame
   */
  get width() {
    return this.bounds.width;
  }

  /**
   * Returns the height of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {number} - In the root coordinate frame
   */
  get height() {
    return this.bounds.height;
  }

  /**
   * Returns the x of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {number} - In the root coordinate frame
   */
  get x() {
    return this.trail.getParentTransform().transformX( this.node.x );
  }

  /**
   * Sets the x of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {number} value - In the root coordinate frame
   */
  set x( value ) {
    this.node.x = this.trail.getParentTransform().inverseX( value );
  }

  /**
   * Returns the y of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {number} - In the root coordinate frame
   */
  get y() {
    return this.trail.getParentTransform().transformY( this.node.y );
  }

  /**
   * Sets the y of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {number} value - In the root coordinate frame
   */
  set y( value ) {
    this.node.y = this.trail.getParentTransform().inverseY( value );
  }

  /**
   * Returns the translation of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Vector2} - In the root coordinate frame
   */
  get translation() {
    return this.trail.getParentTransform().transformPosition2( this.node.translation );
  }

  /**
   * Sets the translation of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {Vector2} value - In the root coordinate frame
   */
  set translation( value ) {
    this.node.translation = this.trail.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the left of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {number} - In the root coordinate frame
   */
  get left() {
    return this.trail.getParentTransform().transformX( this.node.left );
  }

  /**
   * Sets the left of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {number} value - In the root coordinate frame
   */
  set left( value ) {
    this.node.left = this.trail.getParentTransform().inverseX( value );
  }

  /**
   * Returns the right of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {number} - In the root coordinate frame
   */
  get right() {
    return this.trail.getParentTransform().transformX( this.node.right );
  }

  /**
   * Sets the right of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {number} value - In the root coordinate frame
   */
  set right( value ) {
    this.node.right = this.trail.getParentTransform().inverseX( value );
  }

  /**
   * Returns the centerX of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {number} - In the root coordinate frame
   */
  get centerX() {
    return this.trail.getParentTransform().transformX( this.node.centerX );
  }

  /**
   * Sets the centerX of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {number} value - In the root coordinate frame
   */
  set centerX( value ) {
    this.node.centerX = this.trail.getParentTransform().inverseX( value );
  }

  /**
   * Returns the top of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {number} - In the root coordinate frame
   */
  get top() {
    return this.trail.getParentTransform().transformY( this.node.top );
  }

  /**
   * Sets the top of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {number} value - In the root coordinate frame
   */
  set top( value ) {
    this.node.top = this.trail.getParentTransform().inverseY( value );
  }

  /**
   * Returns the bottom of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {number} - In the root coordinate frame
   */
  get bottom() {
    return this.trail.getParentTransform().transformY( this.node.bottom );
  }

  /**
   * Sets the bottom of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {number} value - In the root coordinate frame
   */
  set bottom( value ) {
    this.node.bottom = this.trail.getParentTransform().inverseY( value );
  }

  /**
   * Returns the centerY of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {number} - In the root coordinate frame
   */
  get centerY() {
    return this.trail.getParentTransform().transformY( this.node.centerY );
  }

  /**
   * Sets the centerY of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {number} value - In the root coordinate frame
   */
  set centerY( value ) {
    this.node.centerY = this.trail.getParentTransform().inverseY( value );
  }

  /**
   * Returns the leftTop of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Vector2} - In the root coordinate frame
   */
  get leftTop() {
    return this.trail.getParentTransform().transformPosition2( this.node.leftTop );
  }

  /**
   * Sets the leftTop of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {Vector2} value - In the root coordinate frame
   */
  set leftTop( value ) {
    this.node.leftTop = this.trail.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the centerTop of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Vector2} - In the root coordinate frame
   */
  get centerTop() {
    return this.trail.getParentTransform().transformPosition2( this.node.centerTop );
  }

  /**
   * Sets the centerTop of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {Vector2} value - In the root coordinate frame
   */
  set centerTop( value ) {
    this.node.centerTop = this.trail.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightTop of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Vector2} - In the root coordinate frame
   */
  get rightTop() {
    return this.trail.getParentTransform().transformPosition2( this.node.rightTop );
  }

  /**
   * Sets the rightTop of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {Vector2} value - In the root coordinate frame
   */
  set rightTop( value ) {
    this.node.rightTop = this.trail.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the leftCenter of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Vector2} - In the root coordinate frame
   */
  get leftCenter() {
    return this.trail.getParentTransform().transformPosition2( this.node.leftCenter );
  }

  /**
   * Sets the leftCenter of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {Vector2} value - In the root coordinate frame
   */
  set leftCenter( value ) {
    this.node.leftCenter = this.trail.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the center of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Vector2} - In the root coordinate frame
   */
  get center() {
    return this.trail.getParentTransform().transformPosition2( this.node.center );
  }

  /**
   * Sets the center of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {Vector2} value - In the root coordinate frame
   */
  set center( value ) {
    this.node.center = this.trail.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightCenter of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Vector2} - In the root coordinate frame
   */
  get rightCenter() {
    return this.trail.getParentTransform().transformPosition2( this.node.rightCenter );
  }

  /**
   * Sets the rightCenter of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {Vector2} value - In the root coordinate frame
   */
  set rightCenter( value ) {
    this.node.rightCenter = this.trail.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the leftBottom of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Vector2} - In the root coordinate frame
   */
  get leftBottom() {
    return this.trail.getParentTransform().transformPosition2( this.node.leftBottom );
  }

  /**
   * Sets the leftBottom of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {Vector2} value - In the root coordinate frame
   */
  set leftBottom( value ) {
    this.node.leftBottom = this.trail.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the centerBottom of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Vector2} - In the root coordinate frame
   */
  get centerBottom() {
    return this.trail.getParentTransform().transformPosition2( this.node.centerBottom );
  }

  /**
   * Sets the centerBottom of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {Vector2} value - In the root coordinate frame
   */
  set centerBottom( value ) {
    this.node.centerBottom = this.trail.getParentTransform().inversePosition2( value );
  }

  /**
   * Returns the rightBottom of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @returns {Vector2} - In the root coordinate frame
   */
  get rightBottom() {
    return this.trail.getParentTransform().transformPosition2( this.node.rightBottom );
  }

  /**
   * Sets the rightBottom of the last node in the trail, but in the root coordinate frame.
   * @public
   *
   * @param {Vector2} value - In the root coordinate frame
   */
  set rightBottom( value ) {
    this.node.rightBottom = this.trail.getParentTransform().inversePosition2( value );
  }

  /**
   * Releases references, and frees it to the pool.
   * @public
   */
  dispose() {
    this.trail = null;

    // for now
    this.freeToPool();
  }
}

scenery.register( 'LayoutProxy', LayoutProxy );

Poolable.mixInto( LayoutProxy );

export default LayoutProxy;