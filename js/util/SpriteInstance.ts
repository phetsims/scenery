// Copyright 2019-2023, University of Colorado Boulder

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
import Pool from '../../../phet-core/js/Pool.js';
import { scenery, Sprite } from '../imports.js';

const scratchVector = new Vector2( 0, 0 );
const scratchMatrix = Matrix3.IDENTITY.copy();

export class SpriteInstanceTransformType extends EnumerationValue {
  public static readonly TRANSLATION = new SpriteInstanceTransformType();
  public static readonly TRANSLATION_AND_SCALE = new SpriteInstanceTransformType();
  public static readonly TRANSLATION_AND_ROTATION = new SpriteInstanceTransformType();
  public static readonly AFFINE = new SpriteInstanceTransformType();

  public static readonly enumeration = new Enumeration( SpriteInstanceTransformType, {
    phetioDocumentation: 'Defines the available transform type for a SpriteInstance'
  } );
}

export default class SpriteInstance {

  // This should be set to a `Sprite` object which is the sprite that should be displayed.
  // This field is expected to be set by the client whenever it needs to change.
  public sprite: Sprite | null;

  // Please just mutate the given Matrix3 for performance. If the matrix represents something
  // other than just a translation, please update the `transformType` to the type that represents the possible
  // values.
  public matrix: Matrix3;

  public transformType: SpriteInstanceTransformType;

  // The general opacity/alpha of the displayed sprite (see Node's opacity)
  public alpha: number;

  public constructor() {

    this.sprite = null;
    this.matrix = new Matrix3().setToAffine( 1, 0, 0, 0, 1, 0 ); // initialized to trigger the affine flag
    this.transformType = SpriteInstanceTransformType.TRANSLATION;

    this.alpha = 1;
  }

  /**
   * For pooling. Please use SpriteInstance.dirtyFromPool() to grab a copy
   */
  public initialize(): this {
    // We need an empty initialization method here, so that we can grab dirty versions and use them for higher
    // performance.

    return this;
  }

  /**
   * Returns a Shape that represents the hit-testable area of this SpriteInstance.
   */
  public getShape(): Shape {
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
  public containsPoint( point: Vector2 ): boolean {
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

  public freeToPool(): void {
    SpriteInstance.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( SpriteInstance, {
    maxSize: 1000
  } );
}

scenery.register( 'SpriteInstance', SpriteInstance );
