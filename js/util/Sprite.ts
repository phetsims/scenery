// Copyright 2019-2022, University of Colorado Boulder

/**
 * Represents a single sprite for the Sprites node, whose image can change over time (if it gets regenerated, etc.).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import IProperty from '../../../axon/js/IProperty.js';
import Property from '../../../axon/js/Property.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Shape from '../../../kite/js/Shape.js';
import { scenery, SpriteImage } from '../imports.js';

class Sprite {

  imageProperty: IProperty<SpriteImage>;

  constructor( spriteImage: SpriteImage ) {
    assert && assert( spriteImage instanceof SpriteImage );

    this.imageProperty = new Property( spriteImage );
  }

  /**
   * Returns a Shape that represents the hit-testable area of this Sprite.
   */
  getShape(): Shape {
    return this.imageProperty.value.getShape();
  }

  /**
   * Returns whether a given point is considered "inside" the Sprite
   */
  containsPoint( point: Vector2 ): boolean {
    return this.imageProperty.value.containsPoint( point );
  }
}

scenery.register( 'Sprite', Sprite );
export default Sprite;