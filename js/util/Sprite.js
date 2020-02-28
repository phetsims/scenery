// Copyright 2019-2020, University of Colorado Boulder

/**
 * Represents a single sprite for the Sprites node, whose image can change over time (if it gets regenerated, etc.).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import scenery from '../scenery.js';
import SpriteImage from './SpriteImage.js';

class Sprite {
  /**
   * @param {SpriteImage} spriteImage - The initial SpriteImage
   */
  constructor( spriteImage ) {
    assert && assert( spriteImage instanceof SpriteImage );

    // @public {Property.<SpriteImage>}
    this.imageProperty = new Property( spriteImage );
  }
}

scenery.register( 'Sprite', Sprite );
export default Sprite;