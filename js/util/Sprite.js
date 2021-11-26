// Copyright 2019-2021, University of Colorado Boulder

/**
 * Represents a single sprite for the Sprites node, whose image can change over time (if it gets regenerated, etc.).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import { scenery, SpriteImage } from '../imports.js';

class Sprite {
  /**
   * @param {SpriteImage} spriteImage - The initial SpriteImage
   */
  constructor( spriteImage ) {
    assert && assert( spriteImage instanceof SpriteImage );

    // @public {Property.<SpriteImage>}
    this.imageProperty = new Property( spriteImage );
  }

  /**
   * Returns a Shape that represents the hit-testable area of this Sprite.
   * @public
   *
   * @returns {Shape}
   */
  getShape() {
    return this.imageProperty.value.getShape();
  }

  /**
   * Returns whether a given point is considered "inside" the Sprite
   * @public
   *
   * @param {Vector2} point
   * @returns {boolean}
   */
  containsPoint( point ) {
    return this.imageProperty.value.containsPoint( point );
  }
}

scenery.register( 'Sprite', Sprite );
export default Sprite;