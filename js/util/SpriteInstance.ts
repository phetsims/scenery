// Copyright 2019-2022, University of Colorado Boulder

/**
 * Represents a single instance on the screen of a given Sprite object. It can have its own transformation matrix, and
 * is set up as a lightweight container of information, for high-performance usage (with pooling).
 *
 * Its individual parameters should generally be mutated directly by the client.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';
import EnumerationValue from '../../../phet-core/js/EnumerationValue.js';
import Poolable, { PoolableVersion } from '../../../phet-core/js/Poolable.js';
import { scenery, Sprite } from '../imports.js';

const scratchVector = new Vector2( 0, 0 );
const scratchMatrix = Matrix3.IDENTITY.copy();

class SpriteInstanceTransformType extends EnumerationValue {
  static TRANSLATION = new SpriteInstanceTransformType();
  static TRANSLATION_AND_SCALE = new SpriteInstanceTransformType();
  static TRANSLATION_AND_ROTATION = new SpriteInstanceTransformType();
  static AFFINE = new SpriteInstanceTransformType();

  static enumeration = new Enumeration( SpriteInstanceTransformType, {
    phetioDocumentation: 'Defines the available transform type for a SpriteInstance'
  } );
}

class SpriteInstance {

  // This should be set to a `Sprite` object which is the sprite that should be displayed.
  // This field is expected to be set by the client whenever it needs to change.
  sprite: Sprite | null;

  // Please just mutate the given Matrix3 for performance. If the matrix represents something
  // other than just a translation, please update the `transformType` to the type that represents the possible
  // values.
  matrix: Matrix3;

  transformType: SpriteInstanceTransformType;

  // The general opacity/alpha of the displayed sprite (see Node's opacity)
  alpha: number;

  constructor() {

    this.sprite = null;
    this.matrix = new Matrix3().setToAffine( 1, 0, 0, 0, 1, 0 ); // initialized to trigger the affine flag
    this.transformType = SpriteInstanceTransformType.TRANSLATION;

    this.alpha = 1;
  }

  /**
   * For pooling. Please use SpriteInstance.dirtyFromPool() to grab a copy
   */
  private initialize() {
    // We need an empty initialization method here, so that we can grab dirty versions and use them for higher
    // performance.
  }

  /**
   * Returns a Shape that represents the hit-testable area of this SpriteInstance.
   */
  getShape(): Shape {
    if ( this.sprite ) {
      return this.sprite.getShape().transformed( this.matrix );
    }
    else {
      return new Shape();
    }
  }

  /**
   * Returns whether a given point is considered "inside" the SpriteInstance
   */
  containsPoint( point: Vector2 ): boolean {
    if ( !this.sprite ) {
      return false;
    }

    const position = scratchVector.set( point );

    if ( this.transformType === SpriteInstanceTransformType.AFFINE ) {
      scratchMatrix.set( this.matrix ).invert().multiplyVector2( position );
    }
    else {
      position.x -= this.matrix.m02();
      position.y -= this.matrix.m12();

      if ( this.transformType === SpriteInstanceTransformType.TRANSLATION_AND_SCALE ) {
        position.x /= this.matrix.m00();
        position.y /= this.matrix.m11();
      }
      else if ( this.transformType === SpriteInstanceTransformType.TRANSLATION_AND_ROTATION ) {
        position.rotate( -this.matrix.rotation );
      }
    }

    return this.sprite.containsPoint( position );
  }
}

type PoolableSpriteInstance = PoolableVersion<typeof SpriteInstance>;
const PoolableSpriteInstance = Poolable.mixInto( SpriteInstance, { // eslint-disable-line
  maxSize: 1000
} );

scenery.register( 'SpriteInstance', SpriteInstance );
export default PoolableSpriteInstance;
export { SpriteInstanceTransformType, SpriteInstance as RawSpriteInstance };