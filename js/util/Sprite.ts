// Copyright 2019-2022, University of Colorado Boulder

/**
 * Represents a single sprite for the Sprites node, whose image can change over time (if it gets regenerated, etc.).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TProperty from '../../../axon/js/TProperty.js';
import Property from '../../../axon/js/Property.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { Shape } from '../../../kite/js/imports.js';
import { scenery, SpriteImage } from '../imports.js';

export default class Sprite {

  public readonly imageProperty: TProperty<SpriteImage>;

  public constructor( spriteImage: SpriteImage ) {
    this.imageProperty = new Property( spriteImage );
  }

  /**
   * Returns a Shape that represents the hit-testable area of this Sprite.
   */
  public getShape(): Shape {
    return this.imageProperty.value.getShape();
  }

  /**
   * Returns whether a given point is considered "inside" the Sprite
   */
  public containsPoint( point: Vector2 ): boolean {
    return this.imageProperty.value.containsPoint( point );
  }
}

scenery.register( 'Sprite', Sprite );
